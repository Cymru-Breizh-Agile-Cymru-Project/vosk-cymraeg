from text_process.normalise import cleanup_utf8_chars

VALID_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÂÊÎÔÛŴŶÏÖ abcdefghijklmnopqrstuvwxyzâêîôûŵŷï'-<>_áéöòàäë"
)


def normalise_sentence(s: str) -> str:
    for special_tag in [
        "<anadlu i mewn yn sydyn>",
        "<chwythu allan>",
        "<clirio gwddf>",
    ]:
        if special_tag in s:
            s = s.replace(special_tag, special_tag.replace(" ", "_"))

    # Normalize apostrophes
    s = cleanup_utf8_chars(s)
    s = (
        s.replace("*", "")
        .replace("[anadl]", "<anadlu>")
        .replace("<chwerthin)", "<chwerthin>")
        .replace("{aneglur}", "<aneglur>")
        .replace("{chwerthin}", "<chwerthin>")
        .replace("–", "-")
        .replace("¬", "-")
        .replace("—", "-")
        .replace("/", "")
        .replace("''", "'")
    )

    s = remove_punctuation(s).strip()
    return s.lower()  # We could preserve capitalized words in the future


def remove_punctuation(sentence: str) -> str:
    kept_chars = []
    for c in sentence:
        if c in ',.?!…;:"':
            continue
        kept_chars.append(c)
    cleaned = "".join(kept_chars)
    return " ".join(cleaned.split())  # Remove multi-spaces


def get_non_domain_chars(sentence: str) -> set:
    chars = set(sentence)
    return chars.difference(VALID_CHARS)
