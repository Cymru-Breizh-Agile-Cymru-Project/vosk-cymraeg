import re
from io import StringIO
from pathlib import Path

import polars as pl
import requests

from vosk_cymraeg.phonetics.llef_py3 import get_unstressed_phones


class Phonemizer:
    def __init__(self):
        """Loads Geiriadur Ynganu Bangor into memory and a pronunciation loopup table for llef_py3.py"""
        PATTERN = re.compile("([^ ]+) (.+) (/.*/)")
        text = Path("data/external/geiriadur-ynganu-bangor/bangordict.dict").read_text()
        words = []
        for line in text.splitlines():
            res = PATTERN.fullmatch(line)
            words.append(
                (
                    res.group(1),
                    res.group(2).replace("'", "").replace("-", "").split(),
                    res.group(3),
                )
            )
        self._table = pl.DataFrame(
            words, orient="row", schema=["Word", "Pronunciation", "IPA"]
        )

        # Create lookup table
        r = requests.get(
            "https://docs.google.com/spreadsheets/d/1LekYLxMiBT3kRFxuNQPPXqAl2MZkhSqpwC4wUHxsVVo/export?gid=0&format=tsv"
        )
        r.encoding = r.apparent_encoding  # Fix encoding
        lookup_table = pl.read_csv(StringIO(r.text), separator="\t").drop(
            ["Notes", "Geriadur-ynganu-bangor equivalent"]
        )
        self._lookup_dict = {key: value for (key, value) in lookup_table.rows()}

    def phonemize(self, word: str) -> list[list[str]]:
        res = self._table.filter(pl.col("Word") == word)
        if len(res):
            return res["Pronunciation"].to_list()

        try:
            return [
                [
                    self._lookup_dict.get(phone, phone)
                    for phone in get_unstressed_phones(word.lower())[0]
                ]
            ]
        except TypeError:
            return []
