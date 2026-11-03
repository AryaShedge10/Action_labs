"""Lightweight text utilities (no external NLP dependencies)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "approx", "prof", "jr", "sr"}
_TAG_QUESTION = re.compile(
    r",?\s*(?:okay|ok|alright|all right|right|sound good|does that sound good|does that make sense)\s*\?\s*$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    text: str
    is_question: bool


def split_sentences(text: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        word = re.search(r"([A-Za-z]+)\s*$", text[start:match.start()])
        if match.group() == "." and word and word.group(1).lower() in _ABBREVIATIONS:
            continue
        sentences.append(_make_sentence(text, start, match.end()))
        start = match.end()
    if text[start:].strip():
        sentences.append(_make_sentence(text, start, len(text)))
    return sentences


def _make_sentence(text: str, start: int, end: int) -> Sentence:
    # Trim leading whitespace but keep offsets relative to the original text.
    while start < end and text[start].isspace():
        start += 1
    chunk = text[start:end]
    is_question = chunk.rstrip().endswith("?") and not _TAG_QUESTION.search(chunk)
    return Sentence(start, end, chunk, is_question)


# --------------------------------------------------------------------------- tokens

STOPWORDS = frozenset("""
a an the and or but if of to in on at for from by with without about as is are was were be been being
am do does did have has had having i me my we our you your he she it its they them their this that these
those there here so very just also too some any can could would should will shall may might must not no
yes yeah okay ok well like really get got go going let lets let's i'm i've i'll you're it's that's what
which who whom when where why how all both each few more most other such only own same than then now
""".split())

# Words that structure a SOAP claim but carry no clinical content of their own.
SCAFFOLD_WORDS = frozenset("""
patient patients reports reported report states stated denies denied endorses complains presents noted
note notes history current currently medication medications allergy allergies allergic plan planned
assessment clinician impression recommend recommended prescribe prescribed start started continue
examination exam performed documented reviewed physician review pending safety include includes including
mg reaction family member sex recorded form address year years old aged since resolved previous previously
took taking longer possible uncertain unlikely considers confirmed deferred avoid worsening noted consultation
given stated
""".split())

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "a couple": 2, "couple": 2,
}

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token[:-1] if len(token) > 3 and token.endswith("e") else token


def content_stems(text: str, drop_scaffold: bool = False) -> set[str]:
    out = set()
    for tok in tokens(text):
        if tok in STOPWORDS or (drop_scaffold and tok in SCAFFOLD_WORDS):
            continue
        if tok.replace(".", "").isdigit():
            continue  # numbers are verified separately
        out.add(stem(tok))
    return out


def numbers_in(text: str) -> set[str]:
    """All numeric values in `text`, digits and spelled-out, normalised to strings ('3', '101.2')."""
    found = {_norm_number(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}
    lowered = text.lower()
    for word, value in WORD_NUMBERS.items():
        if len(word) > 2 and re.search(rf"\b{re.escape(word)}\b", lowered):
            found.add(str(value))
    return found


def _norm_number(raw: str) -> str:
    return raw.rstrip("0").rstrip(".") if "." in raw else raw.lstrip("0") or "0"


def oxford_join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
