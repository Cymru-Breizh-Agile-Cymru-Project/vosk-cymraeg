from typing import Optional

from vosk_cymraeg.llef_py3 import get_unstressed_phones


def phonemize(word: str) -> Optional[tuple]:
    try:
        phones = get_unstressed_phones(word)[0]
    except:
        return None
    
    # get_unstressed_phones can return a tuple with an empty first element
    if phones[0]:
        return phones
    
    return None