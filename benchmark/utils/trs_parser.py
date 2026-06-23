import os
from lxml import etree


def parse_trs(trs_path: str) -> list:
    try:
        tree = etree.parse(trs_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse TRS file {trs_path}: {e}")

    segments = []
    SKIP_TAGS = {"Sync", "Event", "Who", "Turn", "Section", "Episode"}

    for turn in root.findall(".//Turn"):
        start = float(turn.get("startTime", 0))
        end = float(turn.get("endTime", 0))
        speaker = turn.get("speaker", "UNKNOWN")

        text_parts = []
        if turn.text and turn.text.strip():
            text_parts.append(turn.text.strip())

        for elem in turn:
            if elem.tag not in SKIP_TAGS:
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                text_parts.append(elem.tail.strip())

        text = " ".join(text_parts).strip()
        if not text or (end - start) < 0.1:
            continue

        segments.append({
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        })

    segments.sort(key=lambda x: x["start"])
    return segments


def find_trs(file_id: str, trs_dir: str) -> str | None:
    # Direct match first
    path = os.path.join(trs_dir, f"{file_id}.trs")
    if os.path.exists(path):
        return path
    # WAV/RTTM use -avd suffix, TRS files use -std suffix
    trs_id = file_id.replace("-avd", "-std")
    path = os.path.join(trs_dir, f"{trs_id}.trs")
    return path if os.path.exists(path) else None
