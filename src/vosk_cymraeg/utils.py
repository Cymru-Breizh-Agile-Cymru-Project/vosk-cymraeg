

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
