import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FIELD_COLUMNS = (
    "x_m",
    "y_m",
    "z_m",
    "density_kg_m3",
    "pressure_pa",
    "temperature_c",
    "turbulence_eddy_dissipation_m2_s3",
    "turbulence_kinetic_energy_m2_s2",
    "velocity_u_m_s",
    "velocity_v_m_s",
    "velocity_w_m_s",
)
SOURCE_HEADERS = (
    "X [ m ]",
    "Y [ m ]",
    "Z [ m ]",
    "Density [ kg m^-3 ]",
    "Pressure [ Pa ]",
    "Temperature [ C ]",
    "Turbulence Eddy Dissipation [ m^2 s^-3 ]",
    "Turbulence Kinetic Energy [ m^2 s^-2 ]",
    "Velocity u [ m s^-1 ]",
    "Velocity v [ m s^-1 ]",
    "Velocity w [ m s^-1 ]",
)
CASE_PATTERN = re.compile(
    r"^(?P<configuration>[A-Za-z]+)_P(?P<power>\d+(?:\.\d+)?)_"
    r"(?P<pressure>\d+(?:\.\d+)?)_bar_(?P<mass_flow>\d+(?:\.\d+)?)kgs_"
    r"T_(?P<temperature>-?\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
NULL_VALUES = {"", "null", "none", "nan"}
ROWS_PER_BATCH = 50_000
SOLID_DENSITY_THRESHOLD_KG_M3 = 2_000.0


def parse_case_metadata(path):
    match = CASE_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Filename does not match the expected case pattern: {path.name}")
    values = match.groupdict()
    configuration = values["configuration"].upper()
    if path.parent.name.upper() != configuration:
        raise ValueError(
            f"Folder configuration {path.parent.name!r} does not match "
            f"filename configuration {configuration!r}: {path}"
        )
    return {
        "case_id": path.stem,
        "configuration": configuration,
        "power_parameter_value": float(values["power"]),
        "pressure_bar": float(values["pressure"]),
        "mass_flow_kg_s": float(values["mass_flow"]),
        "temperature_parameter_value": 305.0 if float(values["temperature"]) == 205.0 else float(values["temperature"]),
    }


def read_case(path):
    pending = []
    cleaned_batches = []
    seen_coordinates = {}
    plane_ids = set()
    plane_section_count = 0
    counts = {
        "raw_rows": 0,
        "malformed_rows": 0,
        "incomplete_rows": 0,
        "solid_rows": 0,
        "duplicate_coordinate_rows": 0,
        "duplicate_coordinate_conflicts": 0,
    }
    minima = np.full(3, np.inf)
    maxima = np.full(3, -np.inf)

    def flush_pending():
        nonlocal pending
        if not pending:
            return
        kept = []
        kept_planes = []
        for plane_id, raw_values in pending:
            counts["raw_rows"] += 1
            if len(raw_values) != len(FIELD_COLUMNS):
                counts["malformed_rows"] += 1
                continue
            values = []
            malformed = False
            for value in raw_values:
                if value.lower() in NULL_VALUES:
                    values.append(math.nan)
                    continue
                try:
                    number = float(value)
                except ValueError:
                    malformed = True
                    break
                values.append(number)
            if malformed:
                counts["malformed_rows"] += 1
                continue
            if math.isfinite(values[3]) and values[3] > SOLID_DENSITY_THRESHOLD_KG_M3:
                counts["solid_rows"] += 1
                continue
            if not all(math.isfinite(value) for value in values):
                counts["incomplete_rows"] += 1
                continue
            coordinate = (values[0], values[1], values[2])
            first_values = seen_coordinates.get(coordinate)
            if first_values is not None:
                counts["duplicate_coordinate_rows"] += 1
                if first_values != tuple(values[3:]):
                    counts["duplicate_coordinate_conflicts"] += 1
                continue
            seen_coordinates[coordinate] = tuple(values[3:])
            minima[:] = np.minimum(minima, values[:3])
            maxima[:] = np.maximum(maxima, values[:3])
            kept.append(values)
            kept_planes.append(plane_id)
        if kept:
            frame = pd.DataFrame(kept, columns=FIELD_COLUMNS)
            frame.insert(0, "plane_id", kept_planes)
            cleaned_batches.append(frame)
        pending = []

    state = "seeking_name"
    current_plane = None
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for line in source:
            text = line.strip()
            if not text:
                continue
            if text == "[Name]":
                state = "reading_name"
                continue
            if text == "[Data]":
                state = "reading_header"
                continue
            if state == "reading_name":
                current_plane = text.partition(" in Case ")[0].strip()
                plane_ids.add(current_plane)
                plane_section_count += 1
                state = "seeking_data"
                continue
            if state == "reading_header":
                header = next(csv.reader([text], skipinitialspace=True))
                normalized = tuple(item.strip() for item in header)
                if normalized != SOURCE_HEADERS:
                    raise ValueError(
                        f"Unexpected field schema in {path.name}: {normalized!r}"
                    )
                state = "reading_rows"
                continue
            if state == "reading_rows":
                row = next(csv.reader([text], skipinitialspace=True))
                pending.append((current_plane, [value.strip() for value in row]))
                if len(pending) >= ROWS_PER_BATCH:
                    flush_pending()
    flush_pending()

    if not cleaned_batches:
        raise ValueError(f"No complete fluid-domain rows remained after cleaning: {path}")
    extents = maxima - minima
    maximum_planar_extent = max(extents[0], extents[1])
    if maximum_planar_extent > 2.0:
        geometry_class = "large"
    elif maximum_planar_extent <= 1.5:
        geometry_class = "small"
    else:
        geometry_class = "intermediate_or_unknown"

    summary = {
        "case_id": path.stem,
        "source_filename": path.name,
        "configuration": path.parent.name.upper(),
        "plane_count": len(plane_ids),
        "plane_section_count": plane_section_count,
        **counts,
        "retained_rows": sum(len(frame) for frame in cleaned_batches),
        "x_min_m": minima[0],
        "x_max_m": maxima[0],
        "x_extent_m": extents[0],
        "y_min_m": minima[1],
        "y_max_m": maxima[1],
        "y_extent_m": extents[1],
        "z_min_m": minima[2],
        "z_max_m": maxima[2],
        "z_extent_m": extents[2],
        "geometry_class": geometry_class,
    }
    return cleaned_batches, summary


def main():
    parser = argparse.ArgumentParser(
        description="Combine and clean the per-case Fluent plane exports into Parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing one configuration folder per case group.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "mega_dataset.parquet",
        help="Output Parquet file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output Parquet and case-summary CSV.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    summary_path = output_path.with_name(f"{output_path.stem}_case_summary.csv")
    temporary_output = output_path.with_name(f"{output_path.name}.partial")
    temporary_summary = summary_path.with_name(f"{summary_path.name}.partial")
    output_files = (output_path, summary_path, temporary_output, temporary_summary)
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    case_files = sorted(input_dir.glob("*/*.csv"))
    if not case_files:
        raise FileNotFoundError(f"No CSV cases found under {input_dir}")
    metadata_by_case = {path: parse_case_metadata(path) for path in case_files}
    summaries = []
    writer = None
    total_rows = 0
    try:
        for case_number, path in enumerate(case_files, start=1):
            batches, summary = read_case(path)
            metadata = metadata_by_case[path]
            for frame in batches:
                for name, value in metadata.items():
                    if name != "case_id":
                        frame[name] = value
                frame["case_id"] = metadata["case_id"]
                frame["geometry_class"] = summary["geometry_class"]
                frame["loop_x_extent_m"] = summary["x_extent_m"]
                frame["loop_y_extent_m"] = summary["y_extent_m"]
                frame = frame[
                    [
                        "case_id",
                        "configuration",
                        "power_parameter_value",
                        "pressure_bar",
                        "mass_flow_kg_s",
                        "temperature_parameter_value",
                        "geometry_class",
                        "loop_x_extent_m",
                        "loop_y_extent_m",
                        "plane_id",
                        *FIELD_COLUMNS,
                    ]
                ]
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_output, table.schema, compression="zstd"
                    )
                writer.write_table(table, row_group_size=ROWS_PER_BATCH)
                total_rows += len(frame)
            summaries.append(summary)
            print(
                f"[{case_number}/{len(case_files)}] {path.name}: "
                f"{summary['retained_rows']:,} rows kept, "
                f"{summary['solid_rows']:,} solid, "
                f"{summary['incomplete_rows']:,} incomplete, "
                f"{summary['duplicate_coordinate_rows']:,} duplicate coordinates"
            )
        if writer is None:
            raise ValueError("No rows were written; all cases were empty after cleaning.")
        writer.close()
        writer = None
        pd.DataFrame(summaries).to_csv(temporary_summary, index=False)
        os.replace(temporary_output, output_path)
        os.replace(temporary_summary, summary_path)
    except Exception:
        if writer is not None:
            writer.close()
        raise

    print(f"Cases: {len(case_files)}")
    print(f"Retained rows: {total_rows:,}")
    print(f"Dataset: {output_path}")
    print(f"Case summary: {summary_path}")


if __name__ == "__main__":
    main()
