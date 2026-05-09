"""
Convert Praat TextGrid files to RTTM format.

Just drop this script into your folder of .TextGrid files and run:

    python textgrid_to_rttm.py

It will find all .TextGrid files in the same directory as the script and convert them.
Output .rttm files are saved alongside the originals by default.

You can also pass explicit paths:

    python textgrid_to_rttm.py file.TextGrid
    python textgrid_to_rttm.py /path/to/folder/
    python textgrid_to_rttm.py /path/to/folder/ --output-dir /path/to/rttm/

Notes:
    - Speaker ID is taken from the tier name (e.g. "c1" → "c1")
    - Intervals with empty text are skipped (silence)
    - Supports multiple tiers (one per speaker)
"""

import argparse
import re
import sys
from pathlib import Path

# Directory where this script lives — used as default input location
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_textgrid(tg_path: str) -> list:
    """
    Parse a Praat .TextGrid file and return a list of speaker segments.

    Each segment dict contains:
        file_id  : str   - stem of the audio filename
        channel  : int   - always 1
        start    : float - segment start time in seconds
        duration : float - segment duration in seconds
        speaker  : str   - tier name used as speaker ID
    """
    text = Path(tg_path).read_text(encoding="utf-8", errors="replace")
    file_id = Path(tg_path).stem

    segments = []

    # Split into tier blocks — each starts with 'item [N]:'
    tier_blocks = re.split(r'item\s*\[\d+\]\s*:', text)

    for block in tier_blocks[1:]:  # skip header before first item
        # Get tier name
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not name_match:
            continue
        speaker = name_match.group(1).strip()

        # Find all intervals
        interval_blocks = re.split(r'intervals\s*\[\d+\]\s*:', block)
        for interval in interval_blocks[1:]:
            xmin_match = re.search(r'xmin\s*=\s*([\d.]+)', interval)
            xmax_match = re.search(r'xmax\s*=\s*([\d.]+)', interval)
            text_match = re.search(r'text\s*=\s*"([^"]*)"', interval, re.DOTALL)

            if not (xmin_match and xmax_match and text_match):
                continue

            content = text_match.group(1).strip()
            if not content:
                continue  # skip silence/empty intervals

            start    = float(xmin_match.group(1))
            end      = float(xmax_match.group(1))
            duration = round(end - start, 5)

            if duration <= 0:
                continue

            segments.append({
                "file_id":  file_id,
                "channel":  1,
                "start":    round(start, 5),
                "duration": duration,
                "speaker":  speaker,
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


def convert_file(tg_path: str, output_dir=None) -> str:
    tg_path = Path(tg_path)
    out_dir = Path(output_dir) if output_dir else tg_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_path = out_dir / (tg_path.stem + ".rttm")
    with open(rttm_path, "w", encoding="utf-8") as f:
        f.write(segments_to_rttm(parse_textgrid(str(tg_path))))
    return str(rttm_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Praat TextGrid files to RTTM format."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "One or more .TextGrid files or directories. "
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

    tg_files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            # Case-insensitive glob for .TextGrid / .textgrid
            found = sorted(
                f for f in p.rglob("*")
                if f.suffix.lower() == ".textgrid"
            )
            if not found:
                print(f"Warning: no .TextGrid files found in {p}", file=sys.stderr)
            tg_files.extend(found)
        elif p.is_file() and p.suffix.lower() == ".textgrid":
            tg_files.append(p)
        else:
            print(f"Warning: skipping '{inp}' (not a .TextGrid file or directory)", file=sys.stderr)

    if not tg_files:
        print("No .TextGrid files to process.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tg_files)} .TextGrid file(s) to convert.\n")
    ok = fail = 0
    for tg_file in tg_files:
        try:
            out = convert_file(str(tg_file), args.output_dir)
            print(f"  ✓  {tg_file.name}  →  {out}")
            ok += 1
        except Exception as e:
            print(f"  ✗  {tg_file.name}  failed: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone: {ok} converted, {fail} failed.")


if __name__ == "__main__":
    main()