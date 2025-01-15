from pathlib import Path

from vosk_cymraeg.datasets.banc_trawsgrifiadau_bangor import (
    fetch_banc_trawsgrifiadau_bangor,
)
from vosk_cymraeg.datasets.common_voice import process_common_voice
from vosk_cymraeg.datasets.lleisiau_arfor import fetch_lleisiau_arfor


def main() -> None:
    process_common_voice(Path("data/raw/cv/cy"), Path("data/processed/cv/cy"))
    fetch_banc_trawsgrifiadau_bangor(Path("data/processed/banc"))
    fetch_lleisiau_arfor(Path("data/processed/lleisiau_arfor"))
