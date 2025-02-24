import argparse
from pathlib import Path
from typing import List

import polars as pl
from rich import print

from vosk_cymraeg.normalisation import get_non_domain_chars, normalise_sentence
from vosk_cymraeg.phonetics.phonemizer import Phonemizer


def main() -> None:
    """Create a training/test corpus for Kaldi"""
    args = _get_args()
    output_folder = Path("data/output")

    # Load merged corpora
    train_dataset = load_dataset(args.train)

    # Strip punctuation from sentences
    sentences = set(train_dataset["sentence"].unique())
    unique_words: set[str] = {word for s in sentences for word in s.split()}

    # We only provide the train dataset to build the text corpus
    build_text_corpus(sentences, output_folder)

    phones = build_lexicon(unique_words, output_folder)

    # nonsilence_phones.txt
    nonsilence_phones_path = output_folder / "local/dict_nosp/nonsilence_phones.txt"
    with open(nonsilence_phones_path, "w", encoding="utf-8") as f:
        for p in sorted(phones):
            f.write(f"{p}\n")

    # silence_phones.txt
    silence_phones_path = output_folder / "local/dict_nosp/silence_phones.txt"
    with open(silence_phones_path, "w", encoding="utf-8") as f:
        f.write("SIL\noov\nSPN\nLAU\nNSN\n")

    # optional_silence.txt
    optional_silence_path = output_folder / "local/dict_nosp/optional_silence.txt"
    with open(optional_silence_path, "w", encoding="utf-8") as f:
        f.write("SIL\n")

    # Build files specific to train/test datasets
    build_dataset("train", train_dataset, output_folder)

    if args.dev:
        build_dataset(
            "dev",
            load_dataset(args.dev),
            output_folder,
        )

    if args.test:
        build_dataset(
            "test",
            load_dataset(args.test),
            output_folder,
        )


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "export",
        description="Script responsible for exporting Kaldi datasets",
    )

    # Used to wipe the output folder before processing
    parser.add_argument(
        "--train",
        default=Path("data/processed/dataset/train.csv"),
        help="Path to training dataset csv file",
        type=Path,
    )
    parser.add_argument(
        "--dev",
        default=Path("data/processed/dataset/dev.csv"),
        help="Path to development dataset csv file",
        type=Path,
    )
    parser.add_argument(
        "--test",
        default=Path("data/processed/dataset/test.csv"),
        help="Path to evaluation dataset csv file",
        type=Path,
    )
    parser.add_argument("--clear", action="store_true", help="Clears the target folder")
    # parser.add_argument("--output", default="output", help="Target folder for the Kaldi dataset", type=Path)

    return parser.parse_args()


def load_dataset(path: Path) -> pl.DataFrame:
    def filter(sentence: str) -> bool:
        if not sentence:
            return False
        invalid_chars = get_non_domain_chars(sentence)
        if invalid_chars:
            print(f'[yellow]Invalid chars [{"".join(invalid_chars)}] "{sentence}"')
            return False
        return True

    return (
        pl.read_csv(path)
        .with_columns(
            pl.col("sentence").map_elements(normalise_sentence, return_dtype=str)
        )
        .filter(pl.col("sentence").map_elements(filter, return_dtype=bool))
    )


def build_text_corpus(sentences: List[str], output_path: Path) -> None:
    """Build the text corpus"""

    output_path = output_path / "local"
    output_path.mkdir(exist_ok=True, parents=True)

    print("Building 'corpus.txt'... ", end="")

    with open(output_path / "corpus.txt", "w", encoding="utf-8") as _f:
        for s in sorted(sentences):
            _f.write(f"{s}\n")

    print("done")


def build_lexicon(words: set[str], output_path: Path) -> None:
    """
    Build a lexicon from a list of words

        Silences            -> SIL
        Spoken noises       -> SPN
        Non-spoken noises   -> NSN
        Laughter            -> LAU
    """

    special_tags = {
        "<anadlu>": "SPN",
        "<anadlu_i_mewn_yn_sydyn>": "SPN",
        "<aneglur>": "SPN",
        "<cerddoriaeth>": "NSN",
        "<chwerthin>": "LAU",
        "<chwibanu>": "SPN",
        "<chwythu_allan>": "SPN",
        "<clapio>": "NSN",
        "<clirio_gwddf>": "SPN",
        "<cusanu>": "SPN",
        "<distawrwydd>": "SIL",
        "<ochneidio>": "SPN",
        "<PII>": "SPN",
        "<peswch>": "SPN",
        "<sniffian>": "SPN",
        "<twtian>": "SPN",
    }

    print("Building 'lexicon.txt'... ", end="")

    output_path = output_path / "local/dict_nosp"
    output_path.mkdir(exist_ok=True, parents=True)

    phone_set = set()
    phonemizer = Phonemizer()
    with open(output_path / "lexicon.txt", "w", encoding="utf-8") as _f:
        # Write special phones
        _f.write("!SIL SIL\n<UNK> SPN\n")
        for tag in sorted(special_tags.keys()):
            _f.write(f"{tag} {special_tags[tag]}\n")

        # Write regular words with corresponding phones
        for word in sorted(words):
            pronunciations = phonemizer.phonemize(word)
            pronunciations = [
                [char for char in pron if char] for pron in pronunciations
            ]
            pronunciations = [pron for pron in pronunciations if pron]
            if len(pronunciations) == 0:
                print(f"[red]Could not phonemize {word!r}")
                continue
            for pronunciation in pronunciations:
                _f.write(f"{word} {' '.join(pronunciation)}\n")
                phone_set.update(pronunciation)
    print("done")
    return phone_set


def build_dataset(name: str, df: pl.DataFrame, output_path: Path) -> None:
    """Generate Kaldi data for one sub-corpus, should be called for each split"""

    dataset_path = output_path / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    print(f"Building '{name}' dataset... ", end="")

    # Build 'text' file
    with open(dataset_path / "text", "w", encoding="utf-8") as _f:
        for row in df.rows(named=True):
            sentence = row["sentence"].strip()
            _f.write(f"{row['utterance']}\t{sentence}\n")

    # Build 'utt2spk'
    with open(dataset_path / "utt2spk", "w", encoding="utf-8") as _f:
        for row in df.sort("utterance").rows(named=True):
            _f.write(f"{row['utterance']}\t{row['speaker']}\n")

    # Build 'wav.scp'
    with open(dataset_path / "wav.scp", "w", encoding="utf-8") as _f:
        for row in df.sort("path").rows(named=True):
            audio_path = Path(row["path"]).resolve()
            _f.write(f"{row['utterance']}\t{audio_path.resolve()}\n")

    print("done")
