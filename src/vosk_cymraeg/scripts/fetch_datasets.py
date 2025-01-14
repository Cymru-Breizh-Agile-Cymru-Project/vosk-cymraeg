from pathlib import Path

from vosk_cymraeg.datasets.banc_trawsgrifiadau_bangor import fetch_banc_trawsgrifiadau_bangor
from vosk_cymraeg.datasets.lleisiau_arfor import fetch_lleisiau_arfor


def main() -> None:
    fetch_banc_trawsgrifiadau_bangor(Path("data/processed/banc"))
    fetch_lleisiau_arfor(Path("data/processed/lleisiau_arfor"))
