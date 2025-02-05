import argparse
import os
from pathlib import Path
from typing import List

import polars as pl
from rich.console import Console

from vosk_cymraeg.utils import (
    remove_punctuation,
    split_sentence,
    get_non_domain_chars,
    red, yellow,
)
from vosk_cymraeg.phonemizer import phonemize



def main() -> None:
    """Create a training/test corpus for Kaldi"""
    args = _get_args()
    console = Console()
    output_folder = Path("data/output")

    # Load merged corpora
    train_dataset = pl.read_csv(args.train)

    # Strip punctuation from sentences
    unique_words = set()
    sentences = set()
    for row in train_dataset.rows():
        sub_sentences = split_sentence(row[2]) # split_sentences does nothing atm
        for sub in sub_sentences:
            # Rename special tags containing spaces
            # so they stay whole during tokenization
            for special_tag in [
                "<anadlu i mewn yn sydyn>",
                "<chwythu allan>",
                "<clirio gwddf>",
            ]:
                if special_tag in sub:
                    sub = sub.replace(special_tag, special_tag.replace(' ', '_'))

            # Normalize apostrophes
            sub = sub.replace('’', "'")
            sub = sub.replace('‘', "'")

            sub = remove_punctuation(sub).strip()
            if not sub:
                continue
            unvalid_chars = get_non_domain_chars(sub)
            if unvalid_chars:
                print(yellow(f"Unvalid chars [{''.join(unvalid_chars)}] \"{sub}\""))
                continue

            sub = sub.lower() # We could preserve capitalized words in the future

            sentences.add(sub)
            unique_words.update(sub.split())
    
    # We only provide the train dataset to build the text corpus
    build_text_corpus(sentences, output_folder)
    
    phones = build_lexicon(unique_words, output_folder)
    
    # nonsilence_phones.txt
    nonsilence_phones_path = output_folder / "local/dict_nosp/nonsilence_phones.txt"
    with open(nonsilence_phones_path, 'w', encoding='utf-8') as f:
        pass
        for p in sorted(phones):
            f.write(f'{p}\n')
    
    # silence_phones.txt
    silence_phones_path  = output_folder / "local/dict_nosp/silence_phones.txt"
    with open(silence_phones_path, 'w', encoding='utf-8') as f:
        f.write(f'SIL\noov\nSPN\nLAU\nNSN\n')
    
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

    if args.dev:
        build_dataset(
            "dev",
            pl.read_csv(args.dev),
            args.dev.parent / "clips",
            output_folder,
        )

    if args.test:
        build_dataset(
            "test",
            pl.read_csv(args.test),
            args.test.parent / "clips",
            output_folder,
        )
    


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "export",
        description="Script responsible for exporting Kaldi datasets",
    )

    # Used to wipe the output folder before processing
    parser.add_argument("--train", help="Path to training dataset csv file", type=Path)
    parser.add_argument("--dev", help="Path to development dataset csv file", type=Path)
    parser.add_argument("--test", help="Path to evaluation dataset csv file", type=Path)
    parser.add_argument("--clear", action="store_true", help="Clears the target folder")
    # parser.add_argument("--output", default="output", help="Target folder for the Kaldi dataset", type=Path)

    return parser.parse_args()



def build_text_corpus(sentences: List[str], output_path: Path) -> None:
    """Build the text corpus"""
    
    output_path = output_path / "local"
    os.makedirs(output_path, exist_ok=True)

    print("Building 'corpus.txt'... ", end='')

    with open(output_path / "corpus.txt", 'w', encoding='utf-8') as _f:
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
        "<anadlu>":                 "SPN",
        "<anadlu_i_mewn_yn_sydyn>": "SPN",
        "<aneglur>":                "SPN",
        "<cerddoriaeth>":           "NSN",
        "<chwerthin>":              "LAU",
        "<chwibanu>":               "SPN",
        "<chwythu_allan>":          "SPN",
        "<clapio>":                 "NSN",
        "<clirio_gwddf>":           "SPN",
        "<cusanu>":                 "SPN",
        "<distawrwydd>":            "SIL",
        "<ochneidio>":              "SPN",
        "<PII>":                    "SPN",
        "<peswch>":                 "SPN",
        "<sniffian>":               "SPN",
        "<twtian>":                 "SPN",
    }

    print("Building 'lexicon.txt'... ", end='')

    output_path = output_path / "local/dict_nosp"
    os.makedirs(output_path, exist_ok=True)

    phone_set = set()

    with open(output_path / "lexicon.txt", 'w', encoding='utf-8') as _f:
            # Write special phones
            _f.write("!SIL SIL\n"
                        "<UNK> SPN\n")
            for tag in sorted(special_tags.keys()):
                _f.write(f"{tag} {special_tags[tag]}\n")
            
            # Write regular words with corresponding phones
            for word in sorted(words):
                phones = phonemize(word)
                if phones:
                    _f.write(f"{word} {' '.join(phones)}\n")
                    phone_set.update(phones)
                else:
                    print(red(f"Could not phonemize '{word}'"))

    print("done")
    return phone_set



def build_dataset(
        name: str,
        df: pl.DataFrame,
        clips_path: Path,
        output_path: Path
    ) -> None:
    """Generate Kaldi data for one sub-corpus, should be called for each split"""

    dataset_path = output_path / name
    os.makedirs(dataset_path, exist_ok=True)

    print(f"Building '{name}' dataset... ", end='')

    # Build 'text' file
    with open(dataset_path / "text", 'w', encoding='utf-8') as _f:
        for row in df.rows():
            sentence = remove_punctuation(row[2]).strip()
            _f.write(f"{row[1]}\t{sentence}\n")
    
    # Build 'utt2spk'
    with open(dataset_path / "utt2spk", 'w', encoding='utf-8') as _f:
        for row in df.sort("utterance").rows():
            _f.write(f"{row[1]}\t{row[0]}\n")
    
    # Build 'wav.scp'
    with open(dataset_path / "wav.scp", 'w', encoding='utf-8') as _f:
        for row in df.sort("utterance").rows():
            audio_path = clips_path / (row[1] + ".wav")
            _f.write(f"{row[1]}\t{audio_path}\n")
    
    print("done")