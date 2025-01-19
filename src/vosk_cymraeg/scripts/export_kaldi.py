import argparse
import os
from pathlib import Path
from typing import List

import polars as pl
from rich.console import Console


def main() -> None:
    """Create a training/test corpus for Kaldi"""
    args = _get_args()
    console = Console()
    output_folder = Path("data/output")

    # Load merged corpora
    train_dataset = pl.read_csv(args.train)
    
    build_text_corpus(output_folder)
    build_lexicon([], output_folder)
    
    # silence_phones.txt
    silence_phones_path  = output_folder / "local/dict_nosp/silence_phones.txt"
    with open(silence_phones_path, 'w', encoding='utf-8') as f:
        f.write(f'SIL\noov\nSPN\nLAU\nNSN\n')
    
    # nonsilence_phones.txt
    nonsilence_phones_path = output_folder / "local/dict_nosp/nonsilence_phones.txt"
    with open(nonsilence_phones_path, 'w', encoding='utf-8') as f:
        pass
        # for p in sorted(phonemes):
        #     f.write(f'{p}\n')
    
    # optional_silence.txt
    optional_silence_path  = output_folder / "local/dict_nosp/optional_silence.txt"
    with open(optional_silence_path, 'w', encoding='utf-8') as f:
        f.write('SIL\n')
    
    # Build files specific to train/test datasets
    build_dataset(
        "train",
        train_dataset,
        args.train.parent / "clips",
        output_folder
    )
    


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "export",
        description="Script responsible for exporting Kaldi datasets",
    )

    # Used to wipe the output folder before processing
    parser.add_argument("--train", help="Path to training dataset csv file", type=Path)
    parser.add_argument("--test", help="Path to evaluation dataset csv file", type=Path)
    parser.add_argument("--clear", action="store_true", help="Clears the target folder")
    # parser.add_argument("--output", default="output", help="Target folder for the Kaldi dataset", type=Path)

    return parser.parse_args()


def build_text_corpus(output_path: Path) -> None:
    """Build the text corpus"""
    pass


def build_lexicon(words: List[str], output_path: Path) -> None:
    """
        Build a lexicon from a list of words
        
            Silences            -> SIL
            Spoken noises       -> SPN
            Non-spoken noises   -> NSN
            Laughter            -> LAU
    """

    special_tags = {
        "<anadlu>":                 "SPN",
        "<anadlu i mewn yn sydyn>": "SPN",
        "<aneglur>":                "SPN",
        "<cerddoriaeth>":           "NSN",
        "<chwerthin>":              "LAU",
        "<chwibanu>":               "SPN",
        "<chwythu allan>":          "SPN",
        "<clapio>":                 "NSN",
        "<clirio gwddf>":           "SPN",
        "<cusanu>":                 "SPN",
        "<distawrwydd>":            "SIL",
        "<ochneidio>":              "SPN",
        "<PII>":                    "SPN",
        "<peswch>":                 "SPN",
        "<sniffian>":               "SPN",
        "<twtian>":                 "SPN",
    }

    output_path = output_path / "local/dict_nosp"
    os.makedirs(output_path, exist_ok=True)

    with open(output_path / "lexicon.txt", 'w', encoding='utf-8') as _f:
            # Write special phones
            _f.write("!SIL SIL\n"
                        "<UNK> SPN\n")
            for tag in sorted(special_tags.keys()):
                 _f.write(f"{tag} {special_tags[tag]}\n")
            
            # Write regular words with phonetization


def build_dataset(
        name: str,
        df: pl.DataFrame,
        clips_path: Path,
        output_path: Path
    ) -> None:
    print(clips_path)
    dataset_path = output_path / name
    os.makedirs(dataset_path, exist_ok=True)

    # Build 'text' file
    with open(dataset_path / "text", 'w', encoding='utf-8') as _f:
        for row in df.rows():
            _f.write(f"{row[1]}\t{row[2]}\n")
    
    # Build 'utt2spk'
    with open(dataset_path / "utt2spk", 'w', encoding='utf-8') as _f:
        for row in df.sort("utterance").rows():
            _f.write(f"{row[1]}\t{row[0]}\n")
    
    # Build 'wav.scp'
    with open(dataset_path / "wav.scp", 'w', encoding='utf-8') as _f:
        for row in df.sort("utterance").rows():
            audio_path = clips_path / (row[1] + ".wav")
            _f.write(f"{row[1]}\t{audio_path}\n")