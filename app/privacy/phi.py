"""PHI pseudonymisation applied to transcript text before it is sent to an LLM.

Even with a local model this limits what can leak through model logs or a misconfigured
endpoint. Deterministic code (rules, grounding) always works on the original text.
Tokens are consistent within one consultation ("Mr. Davis" -> "Mr. [NAME_1]" everywhere).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NOT_NAMES = {"doctor", "dr", "mr", "mrs", "ms", "yes", "no", "not", "okay", "ok", "hi", "hello", "fine", "good",
              "here", "i", "sorry", "just", "really", "so", "well", "allergic", "taking", "going", "feeling"}

_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), 0),
    ("ID", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), 0),                    # Aadhaar-style
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0),                       # SSN-style
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,5}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"), 0),
    ("ID", re.compile(r"\b(?:mrn|medical record(?: number)?|patient id|id number)\s*(?:is|:|#)?\s*([A-Z0-9\-]{4,})", re.I), 1),
    ("DATE", re.compile(r"\b(?:born on|date of birth|dob)\s*(?:is|:)?\s*([^,.;]+)", re.I), 1),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), 0),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|"
                           r"Boulevard|Blvd|Nagar|Marg)\b"), 0),
    ("NAME", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mister|Dr|Doctor)\.?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"), 1),
    ("NAME", re.compile(r"\b(?:my name is|this is|call me|i'?m)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", re.I), 1),
]


@dataclass
class PHIRedactor:
    mapping: dict[str, str] = field(default_factory=dict)     # original -> token
    counts: dict[str, int] = field(default_factory=dict)

    def _token(self, label: str, original: str) -> str:
        if original not in self.mapping:
            self.counts[label] = self.counts.get(label, 0) + 1
            self.mapping[original] = f"[{label}_{self.counts[label]}]"
        return self.mapping[original]

    def redact(self, text: str) -> str:
        for label, regex, group in _PATTERNS:
            def repl(m: re.Match[str]) -> str:
                target = m.group(group)
                if label == "NAME" and target.split()[0].lower() in _NOT_NAMES:
                    return m.group(0)
                token = self._token(label, target)
                if group == 0:
                    return token
                start, end = m.start(group) - m.start(), m.end(group) - m.start()
                whole = m.group(0)
                return whole[:start] + token + whole[end:]
            text = regex.sub(repl, text)
        return text

    @property
    def redaction_count(self) -> int:
        return len(self.mapping)
