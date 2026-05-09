"""
Convert Transcriber (.trs) files to RTTM format.

Just drop this script into your folder of .trs files and run:

    python trs_to_rttm.py

It will find all .trs files in the same directory as the script and convert them.
Output .rttm files are saved alongside the originals by default.

You can also pass explicit paths:

    python trs_to_rttm.py file.trs
    python trs_to_rttm.py /path/to/folder/
    python trs_to_rttm.py /path/to/folder/ --output-dir /path/to/rttm/
"""

import xml.etree.ElementTree as ET
import argparse
import sys
from pathlib import Path

# Directory where this script lives — used as default input location
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_trs(trs_path: str) -> list:
    tree = ET.parse(trs_path)
    root = tree.getroot()

    audio_filename = root.get("audio_filename")
    file_id = Path(audio_filename).stem if audio_filename else Path(trs_path).stem

    segments = []

    for turn in root.iter("Turn"):
        speaker_str = turn.get("speaker", "").strip()
        if not speaker_str:
            continue

        primary_speaker = speaker_str.split()[0]

        turn_start = float(turn.get("startTime", 0))
        turn_end   = float(turn.get("endTime", 0))

        sync_times = [turn_start]
        for sync in turn.findall("Sync"):
            sync_times.append(float(sync.get("time", 0)))
        sync_times.append(turn_end)
        sync_times = sorted(set(sync_times))

        for i in range(len(sync_times) - 1):
            start    = sync_times[i]
            duration = round(sync_times[i + 1] - start, 5)
            if duration <= 0:
                continue
            segments.append({
                "file_id":  file_id,
                "channel":  1,
                "start":    round(start, 5),
                "duration": duration,
                "speaker":  primary_speaker,
            })

    segments.sort(key=lambda s: s["start"])
    return segments


def segments_to_rttm(segments: list) -> str:
    lines = []
    for seg in segments:
        lines.append(
            f"SPEAKER {seg['file_id']} {seg['channel']} "
            f"{seg['start']:.5f} {seg['duration']:.5f} "
            f"<NA> <NA> {seg['speaker']} <NA> <NA>"
        )
    return "\n".join(lines) + "\n"


def convert_file(trs_path: str, output_dir=None) -> str:
    trs_path = Path(trs_path)
    out_dir  = Path(output_dir) if output_dir else trs_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_path = out_dir / (trs_path.stem + ".rttm")
    with open(rttm_path, "w", encoding="utf-8") as f:
        f.write(segments_to_rttm(parse_trs(str(trs_path))))
    return str(rttm_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Transcriber (.trs) files to RTTM format."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "One or more .trs files or directories. "
            "Defaults to the folder this script is in."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write .rttm files into (default: same folder as each input).",
    )
    args = parser.parse_args()

    # Default to the script's own directory if nothing is passed
    inputs = args.inputs if args.inputs else [str(SCRIPT_DIR)]

    trs_files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(p.rglob("*.trs"))
            if not found:
                print(f"Warning: no .trs files found in {p}", file=sys.stderr)
            trs_files.extend(found)
        elif p.is_file() and p.suffix.lower() == ".trs":
            trs_files.append(p)
        else:
            print(f"Warning: skipping '{inp}' (not a .trs file or directory)", file=sys.stderr)

    if not trs_files:
        print("No .trs files to process.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(trs_files)} .trs file(s) to convert.\n")
    ok = fail = 0
    for trs_file in trs_files:
        try:
            out = convert_file(str(trs_file), args.output_dir)
            print(f"  ✓  {trs_file.name}  →  {out}")
            ok += 1
        except Exception as e:
            print(f"  ✗  {trs_file.name}  failed: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone: {ok} converted, {fail} failed.")


if __name__ == "__main__":
    main()