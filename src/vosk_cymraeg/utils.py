from typing import List



def remove_punctuation(sentence: str) -> str:
    kept_chars = []
    for c in sentence:
        if c in ',.?!…;:"':
            continue
        kept_chars.append(c)
    cleaned = ''.join(kept_chars)
    return ' '.join(cleaned.split()) # Remove multi-spaces



def split_sentence(sentence: str) -> List[str]:
    """TODO"""
    return [sentence]



VALID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZÂÊÎÔŴŶÏ abcdefghijklmnopqrstuvwxyzâêîôŵŷï'-"

def get_non_domain_chars(sentence: str) -> set:
    chars = set(sentence)
    return chars.difference(VALID_CHARS)



RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

def red(string: str) -> str:
    return f"{RED}{string}{RESET}"

def green(string: str) -> str:
    return f"{GREEN}{string}{RESET}"

def yellow(string: str) -> str:
    return f"{YELLOW}{string}{RESET}"

def blue(string: str) -> str:
    return f"{BLUE}{string}{RESET}"