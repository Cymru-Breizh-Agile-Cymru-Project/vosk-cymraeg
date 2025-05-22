import logging

import datasets
import polars as pl

_logger = logging.getLogger(__name__)


def load_techiaith_cofnodycynulliad_en_cy() -> pl.DataFrame:
    """Dataset containing translations from the Welsh parliament's website"""
    _logger.info("Loading dataset 'techiaith/cofnodycynulliad_en-cy' from HuggingFace")
    df = datasets.load_dataset(
        "techiaith/cofnodycynulliad_en-cy", split="train"
    ).to_polars()
    return pl.concat(
        [
            pl.DataFrame({"sentence": df["target"], "lang": "cy"}),
            pl.DataFrame({"sentence": df["source"], "lang": "en"}),
        ]
    )


def load_techiaith_legislation_gov_uk_en_cy() -> pl.DataFrame:
    """Dataset containing translations from legislation.gov.uk"""
    _logger.info(
        "Loading dataset 'techiaith/legislation-gov-uk_en-cy' from HuggingFace"
    )
    df = datasets.load_dataset(
        "techiaith/legislation-gov-uk_en-cy", split="train"
    ).to_polars()
    return pl.concat(
        [
            pl.DataFrame({"sentence": df["target"], "lang": "cy"}),
            pl.DataFrame({"sentence": df["source"], "lang": "en"}),
        ]
    )


def load_techiaith_llyw_cymru_en_cy_ogl() -> pl.DataFrame:
    """Dataset containing translations from llyw.cymru"""
    _logger.info("Loading dataset 'techiaith/llyw-cymru-en-cy-ogl' from HuggingFace")
    df = datasets.load_dataset(
        "techiaith/llyw-cymru-en-cy-ogl", split="train"
    ).to_polars()
    return pl.concat(
        [
            pl.DataFrame({"sentence": df["text_cy"], "lang": "cy"}),
            pl.DataFrame({"sentence": df["text_en"], "lang": "en"}),
        ]
    )


def load_str20tbl_tts_prompts_cy_en() -> pl.DataFrame:
    _logger.info("Loading dataset 'str20tbl/tts-prompts-cy-en' from HuggingFace")
    return (
        datasets.load_dataset("str20tbl/tts-prompts-cy-en", split="train")
        .to_polars()
        .rename({"Lang": "lang", "Sentence": "sentence"})
        .select(["sentence", "lang"])
    )

def load_wanasash_brawddegau_enwau_lleoedd() -> pl.DataFrame:
    """Dataset containing sentences with new placenames """
    _logger.info("Loading dataset 'wanasash/brawddegau_enwau_lleoedd' from HuggingFace")
    df = datasets.load_dataset(
        "wanasash/brawddegau_enwau_lleoedd", split="train"
    ).to_polars()
    return pl.DataFrame({"sentence": df["text"]}),
    )
       
