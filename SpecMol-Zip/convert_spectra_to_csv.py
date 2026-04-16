import argparse
import base64
import csv
import struct
import zlib
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "spectrum_csv"


def local_name(tag):
    return tag.split("}", 1)[-1]


def parse_float(value, field_name, source_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} in {source_name}: {value}") from exc


def choose_mgf_spectrum_id(metadata, file_stem, spectrum_index):
    for key in ("TITLE", "SCANS", "FEATURE_ID", "SPECTRUMID", "SPECTRUM_ID"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return f"{file_stem}_{spectrum_index:06d}"


def parse_mgf_file(path):
    spectra = []
    metadata = {}
    peaks = []
    in_ions = False
    spectrum_index = 0

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                in_ions = True
                metadata = {}
                peaks = []
                continue
            if upper == "END IONS":
                if in_ions:
                    spectrum_id = choose_mgf_spectrum_id(metadata, Path(path).stem, spectrum_index)
                    spectra.append(
                        {
                            "spectrum_id": spectrum_id,
                            "peaks": peaks,
                            "metadata": metadata.copy(),
                        }
                    )
                    spectrum_index += 1
                in_ions = False
                continue
            if not in_ions:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().upper()] = value.strip()
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            mz = parse_float(parts[0], "mz", f"{path} line '{line}'")
            intensity = parse_float(parts[1], "intensity", f"{path} line '{line}'")
            peaks.append((mz, intensity))

    return spectra


def decode_mzml_binary_array(binary_text, precision_bits, compressed):
    payload = base64.b64decode(binary_text.encode("ascii"))
    if compressed:
        payload = zlib.decompress(payload)

    if precision_bits == 32:
        item_size = 4
        fmt_char = "f"
    elif precision_bits == 64:
        item_size = 8
        fmt_char = "d"
    else:
        raise ValueError(f"Unsupported mzML precision_bits={precision_bits}")

    if len(payload) % item_size != 0:
        raise ValueError("Decoded mzML binary array has invalid byte length")
    count = len(payload) // item_size
    if count == 0:
        return []
    return list(struct.unpack("<" + fmt_char * count, payload))


def parse_mzml_binary_array(binary_data_array):
    array_kind = None
    precision_bits = None
    compressed = False

    for child in binary_data_array:
        if local_name(child.tag) != "cvParam":
            continue
        name = child.attrib.get("name", "").lower()
        accession = child.attrib.get("accession", "")
        if accession == "MS:1000514" or name == "m/z array":
            array_kind = "mz"
        elif accession == "MS:1000515" or name == "intensity array":
            array_kind = "intensity"
        elif accession == "MS:1000521" or name == "32-bit float":
            precision_bits = 32
        elif accession == "MS:1000523" or name == "64-bit float":
            precision_bits = 64
        elif accession == "MS:1000574" or name == "zlib compression":
            compressed = True

    binary_elem = next((child for child in binary_data_array if local_name(child.tag) == "binary"), None)
    binary_text = (binary_elem.text or "").strip() if binary_elem is not None else ""
    if not binary_text:
        return array_kind, []
    if precision_bits is None:
        precision_bits = 64
    values = decode_mzml_binary_array(binary_text, precision_bits=precision_bits, compressed=compressed)
    return array_kind, values


def parse_mzml_file(path, ms_level_filter=2):
    spectra = []
    spectrum_counter = 0

    for _, elem in ET.iterparse(str(path), events=("end",)):
        if local_name(elem.tag) != "spectrum":
            continue

        spectrum_id = elem.attrib.get("id") or f"{Path(path).stem}_{spectrum_counter:06d}"
        spectrum_counter += 1
        ms_level = None
        mz_values = []
        intensity_values = []

        for child in elem.iter():
            if local_name(child.tag) != "cvParam":
                continue
            accession = child.attrib.get("accession", "")
            name = child.attrib.get("name", "").lower()
            value = child.attrib.get("value", "")
            if accession == "MS:1000511" or name == "ms level":
                try:
                    ms_level = int(float(value))
                except (TypeError, ValueError):
                    ms_level = None

        if ms_level_filter is not None and ms_level != ms_level_filter:
            elem.clear()
            continue

        for child in elem.iter():
            if local_name(child.tag) != "binaryDataArray":
                continue
            array_kind, values = parse_mzml_binary_array(child)
            if array_kind == "mz":
                mz_values = values
            elif array_kind == "intensity":
                intensity_values = values

        if len(mz_values) != len(intensity_values):
            raise ValueError(
                f"mzML spectrum_id={spectrum_id} has mismatched arrays: "
                f"mz={len(mz_values)} intensity={len(intensity_values)}"
            )

        peaks = list(zip(mz_values, intensity_values))
        spectra.append(
            {
                "spectrum_id": spectrum_id,
                "peaks": peaks,
                "metadata": {"ms_level": ms_level},
            }
        )
        elem.clear()

    return spectra


def iter_input_files(input_path):
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = []
    for suffix in ("*.mgf", "*.mzML", "*.mzml"):
        files.extend(sorted(input_path.rglob(suffix)))
    if not files:
        raise ValueError(f"No supported spectrum files found under {input_path}")
    return files


def parse_spectrum_files(input_path, ms_level_filter=2):
    spectra = []
    for path in iter_input_files(input_path):
        suffix = path.suffix.lower()
        if suffix == ".mgf":
            file_spectra = parse_mgf_file(path)
        elif suffix == ".mzml":
            file_spectra = parse_mzml_file(path, ms_level_filter=ms_level_filter)
        else:
            continue
        for spectrum in file_spectra:
            spectrum["source_file"] = str(path)
        spectra.extend(file_spectra)
    if not spectra:
        raise ValueError("No spectra were parsed from the input files")
    return spectra


def load_label_map(label_csv):
    label_map = {}
    with Path(label_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"spectrum_id", "label"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"label csv must contain columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            spectrum_id = str(row["spectrum_id"]).strip()
            label_map[spectrum_id] = {
                "label": row["label"],
                "weight": row.get("weight", ""),
                "split": row.get("split", ""),
            }
    return label_map


def filter_and_sort_peaks(peaks, min_intensity=0.0):
    filtered = [(mz, intensity) for mz, intensity in peaks if intensity >= min_intensity]
    return sorted(filtered, key=lambda item: item[0])


def write_output_csvs(spectra, output_dir, label_map=None, default_label=None, min_intensity=0.0):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    peaks_csv = output_dir / "peaks.csv"
    labels_csv = output_dir / "labels.csv"

    peak_rows = []
    label_rows = []

    for spectrum in spectra:
        spectrum_id = spectrum["spectrum_id"]
        peaks = filter_and_sort_peaks(spectrum["peaks"], min_intensity=min_intensity)
        if not peaks:
            continue

        if label_map is not None:
            if spectrum_id not in label_map:
                if default_label is None:
                    raise ValueError(
                        f"Missing label for spectrum_id={spectrum_id}. "
                        "Provide a label csv or set --default-label."
                    )
                label_value = str(default_label)
                weight_value = ""
                split_value = ""
            else:
                label_value = label_map[spectrum_id]["label"]
                weight_value = label_map[spectrum_id]["weight"]
                split_value = label_map[spectrum_id]["split"]
        else:
            if default_label is None:
                raise ValueError("No label source provided. Use --label-csv or --default-label.")
            label_value = str(default_label)
            weight_value = ""
            split_value = ""

        label_rows.append(
            {
                "spectrum_id": spectrum_id,
                "label": label_value,
                "weight": weight_value,
                "split": split_value,
            }
        )

        for mz, intensity in peaks:
            peak_rows.append(
                {
                    "spectrum_id": spectrum_id,
                    "mz": f"{mz:.10g}",
                    "intensity": f"{intensity:.10g}",
                }
            )

    if not peak_rows:
        raise ValueError("No peaks passed filtering; peaks.csv would be empty")

    with peaks_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spectrum_id", "mz", "intensity"])
        writer.writeheader()
        writer.writerows(peak_rows)

    with labels_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spectrum_id", "label", "weight", "split"])
        writer.writeheader()
        writer.writerows(label_rows)

    return peaks_csv, labels_csv, len(label_rows), len(peak_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label-csv", type=str, default=None)
    parser.add_argument("--default-label", type=float, default=None)
    parser.add_argument("--ms-level", type=int, default=2)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    label_map = load_label_map(args.label_csv) if args.label_csv else None
    spectra = parse_spectrum_files(args.input_path, ms_level_filter=args.ms_level)
    peaks_csv, labels_csv, num_spectra, num_peaks = write_output_csvs(
        spectra=spectra,
        output_dir=args.output_dir,
        label_map=label_map,
        default_label=args.default_label,
        min_intensity=args.min_intensity,
    )

    print(f"[convert] parsed_spectra={len(spectra)}")
    print(f"[convert] written_spectra={num_spectra}")
    print(f"[convert] written_peaks={num_peaks}")
    print(f"[convert] peaks_csv={peaks_csv}")
    print(f"[convert] labels_csv={labels_csv}")


if __name__ == "__main__":
    main()
