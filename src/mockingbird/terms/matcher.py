"""Heuristic term candidates (acronyms / camel-case) for LLM fallback."""
from __future__ import annotations

import re

from mockingbird.terms.glossary import Glossary

_ACRONYM = re.compile(r"\b[A-ZА-Я][A-ZА-Я0-9]{1,7}\b")
_CAMEL = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_MIXED = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
_TECH = re.compile(r"\b[A-Z][0-9][a-z]+\b")

_STOPWORDS = {
    "THE", "AND", "FOR", "WAS", "ARE", "YOU", "OUR", "WITH", "NOT", "CAN", "BUT",
    "HAS", "HAD", "HAVE", "FROM", "THAT", "THIS", "THESE", "THOSE", "WILL",
    "WOULD", "SHOULD", "COULD", "MUST", "MAY", "MIGHT", "YOUR", "THEIR",
    "THERE", "WHERE", "WHICH", "WHAT", "WHEN", "WHO", "WHOM", "WHY", "HOW",
    "ALL", "ANY", "SOME", "MORE", "MOST", "LESS", "THAN", "HERE", "NOW",
    "NEW", "OLD", "TOP", "END", "OUT", "OFF", "OVER", "UNDER", "BEFORE",
    "AFTER", "ONE", "TWO", "THREE", "MAN", "MEN", "WOMAN", "WOMEN", "DAY",
    "DAYS", "WEEK", "MONTH", "YEAR", "YEARS", "GET", "GOT", "GIVE", "GAVE",
    "MAKE", "MADE", "USE", "USED", "USES", "PUT", "RUN", "RAN", "SET", "SEE",
    "SAW", "SAY", "SAID", "SAYS", "COME", "CAME", "GO", "GOES", "WENT",
    "TAKE", "TOOK", "KNOW", "KNEW", "THINK", "THOUGHT", "THING", "THINGS",
    "WAY", "WAYS", "TIME", "TIMES", "GOOD", "BAD", "BIG", "SMALL", "HIGH",
    "LOW", "FAST", "SLOW", "BACK", "FRONT", "LEFT", "RIGHT", "UP", "DOWN",
    "INTO", "ONTO", "DURING", "WITHOUT", "ABOUT", "AGAIN", "ALSO", "EVERY",
    "EACH", "OTHER", "ANOTHER", "FIRST", "LAST", "NEXT", "STILL", "EVEN",
    "JUST", "ONLY", "VERY", "REALLY", "ALWAYS", "NEVER", "OFTEN", "SINCE",
    "UNTIL", "BOTH", "EITHER", "NEITHER", "NOR", "PER", "DOES", "DID",
    "DOING", "IS", "WERE", "BE", "BEEN", "BEING", "AM", "OUR", "ITS",
}


class CandidateExtractor:
    def __init__(self, glossary: Glossary):
        self._known = {e.term.lower() for e in glossary.entries}
        for entry in glossary.entries:
            self._known.update(a.lower() for a in entry.aliases)

    def extract(self, text: str) -> list[str]:
        found: set[str] = set()
        for pattern in (_ACRONYM, _CAMEL, _MIXED, _TECH):
            for m in pattern.finditer(text):
                token = m.group(0)
                if token.upper() in _STOPWORDS:
                    continue
                if token.lower() in self._known:
                    continue
                found.add(token)
        return sorted(found)
