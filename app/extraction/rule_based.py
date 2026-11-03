"""Deterministic clinical entity extractor.

Runs fully offline and is the backbone of the pipeline: the LLM extractor (when available)
can only *add* verified entities on top of this, never silently override it.

Order of passes per sentence:
  1. allergy statements  (spans are consumed so "allergic to amoxicillin" is not a prescription)
  2. vitals              (regex with units)
  3. demographics        (age / sex / form of address)
  4. lexicon terms       (symptoms, findings, diagnoses, procedures, medications) + context cues
Then dialogue-level inference (doctor question -> patient yes/no answer) and aggregation of
mentions into entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.knowledge import drugs as kb
from app.knowledge import terms
from app.models.schemas import (
    CLINICIAN_SPEAKERS,
    ACTIVE_ALLERGY_STATUSES,
    ClinicalEntity,
    EntitySet,
    EntityStatus,
    EntityType,
    Evidence,
    Segment,
    Speaker,
    Transcript,
)
from app.nlp import context as ctxlib
from app.nlp.matcher import LexiconMatcher, TermMatch, clinical_matcher
from app.nlp.text import Sentence, split_sentences

_I = re.IGNORECASE
NKDA = "no known drug allergies"

# --------------------------------------------------------------------------- patterns

_ALLERGY_TRIGGER = re.compile(
    r"\b(?:allergic to|allergy to|allergies to|allergic reaction to|bad reaction to|reaction to|"
    r"react(?:ed|s)? (?:badly )?to|intolerant (?:of|to)|sensitive to)\s+", _I)
_ALLERGY_SUFFIX = re.compile(r"\b(?P<item>[a-z][a-z\-]+(?:\s[a-z][a-z\-]+)?)\s+allerg(?:y|ies)\b", _I)
_ALLERGY_LIST_END = re.compile(
    r"[.;!?]|,\s*(?:but|and i|it|which|so|because|that|i|they)\b|\s(?:but|because|so|which|when|since|"
    r"and i|it gives|it makes|they give|that gives|last time)\b", _I)
_NO_ALLERGY = re.compile(
    r"\b(?:no (?:known )?(?:drug |medication |medicine )?allergies|nkda|nka|not allergic to anything|"
    r"(?:don'?t|do not) have any (?:known |drug )?allergies|no allergies)\b", _I)
_GENERIC_ALLERGY_WORDS = {"drug", "food", "any", "no", "known", "seasonal", "environmental", "medication",
                          "medicine", "other", "severe", "bad", "some", "my", "the", "an", "a", "of", "skin"}
_DRUG_REACTION_POST = re.compile(r"^\s*(?:always\s+|usually\s+)?(?:gives|gave|makes|made|causes|caused)\s+me\b", _I)
_REACTION_BACKREF = re.compile(r"\b(?:it|them|that|last time|when i took|took it|reaction|broke out)\b", _I)
_REACTION_DESCRIPTION = re.compile(r"\b(?:gives?|gave|makes?|made|causes?|caused)\s+me\s+(?:a\s+|an\s+|bad\s+)?$", _I)

_VITAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("temperature", re.compile(
        r"\b(?:temperature|temp|fever)\s*(?:is|was|of|at|reading|reads|:|=)?\s*(?:about|around|up to|of)?\s*"
        r"(?P<v>\d{2,3}(?:\.\d{1,2})?)\s*(?:°|º|degrees?|deg)?\s*(?P<u>fahrenheit|celsius|f|c)?\b", _I)),
    ("blood pressure", re.compile(
        r"\b(?:blood pressure|bp)\s*(?:is|was|of|at|reads|:|=)?\s*(?:about|around)?\s*"
        r"(?P<v>\d{2,3}\s*(?:/|over)\s*\d{2,3})", _I)),
    ("heart rate", re.compile(
        r"\b(?:heart rate|pulse(?: rate)?|hr)\s*(?:is|was|of|at|:|=)?\s*(?:about|around)?\s*(?P<v>\d{2,3})"
        r"(?:\s*(?:bpm|beats))?", _I)),
    ("respiratory rate", re.compile(
        r"\b(?:respiratory rate|resp rate|breathing rate|rr)\s*(?:is|was|of|at|:|=)?\s*(?P<v>\d{1,2})\b", _I)),
    ("oxygen saturation", re.compile(
        r"\b(?:oxygen saturation|o2 sat(?:uration)?s?|sats?|spo2|pulse ox|oxygen(?: level)?)\s*(?:is|was|of|at|:|=)?"
        r"\s*(?:about|around)?\s*(?P<v>\d{2,3})\s*(?:%|percent)?", _I)),
    ("weight", re.compile(
        r"\b(?:weigh(?:s|t)?|weight)\s*(?:is|was|of|:|=|about|around)?\s*(?P<v>\d{2,3}(?:\.\d)?)\s*"
        r"(?P<u>kg|kilos?|kilograms?|lbs?|pounds)\b", _I)),
]
_VITAL_UNITS = {"blood pressure": "mmHg", "heart rate": "bpm", "respiratory rate": "breaths/min",
                "oxygen saturation": "%"}

_AGE_PATTERNS = [
    re.compile(r"\b(?P<age>\d{1,3})[\s\-](?:years?|yrs?)[\s\-]old\b", _I),
    re.compile(r"\b(?:age|aged)\s+(?P<age>\d{1,3})\b", _I),
]
_AGE_SELF = re.compile(
    r"\bI(?:'m| am)\s+(?P<age>\d{1,3})(?:\s+years?(?:\s+old)?)?(?![\d.])"
    r"(?!\s*(?:days?|weeks?|months?|hours?|minutes?|times?|percent|%|mg|pounds?|lbs?|kg|cm|feet|foot|inches|"
    r"degrees|out of|of|weeks? pregnant))", _I)
_SEX_EXPLICIT = [
    re.compile(r"\b\d{1,3}[\s\-]years?[\s\-]old\s+(?P<sex>man|woman|male|female|boy|girl|gentleman|lady)\b", _I),
    re.compile(r"\b(?P<sex>male|female) patient\b", _I),
]
_SEX_SELF = re.compile(r"\bI(?:'m| am) a (?:\d{1,3}[\s\-]years?[\s\-]old )?(?P<sex>man|woman|male|female|guy|boy|girl)\b", _I)
_HONORIFIC = re.compile(r"\b(?P<h>Mr|Mrs|Ms|Miss|Mister)\.?\s+[A-Z][a-z]+")
_SEX_MAP = {"man": "male", "male": "male", "boy": "male", "gentleman": "male", "guy": "male", "mr": "male",
            "mister": "male", "woman": "female", "female": "female", "girl": "female", "lady": "female",
            "mrs": "female", "ms": "female", "miss": "female"}

_DOSE = re.compile(r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|milligrams?|mcg|micrograms?|g|grams?|ml|milliliters?|"
                   r"units?|puffs?|tablets?|tabs?|capsules?|caps?)\b", _I)
_DOSE_BEFORE = re.compile(r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|milligrams?|mcg|g|grams?)\s+(?:of\s+)?$", _I)
_FREQUENCY = re.compile(
    r"\b(once (?:a|per) day|once daily|twice (?:a|per) day|twice daily|three times (?:a|per) day|three times daily|"
    r"four times (?:a|per) day|four times daily|every \d+ hours|every (?:morning|night|evening)|at (?:bedtime|night)|"
    r"daily|nightly|as needed|bid|tid|qid|prn|q\d+h)\b", _I)
_MED_DURATION = re.compile(
    r"\bfor (?:the next |a total of )?(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten|fourteen)\s+"
    r"(?P<u>days?|weeks?|months?)\b", _I)
_ROUTE = re.compile(r"\b(by mouth|orally|oral|topical(?:ly)?|inhaled|inhaler|intravenous(?:ly)?|iv|"
                    r"intramuscular(?:ly)?|injection|nasal spray|eye drops|ear drops|subcutaneous(?:ly)?)\b", _I)
_DOSES_PER_DAY = {"once a day": 1, "once per day": 1, "once daily": 1, "daily": 1, "nightly": 1, "at bedtime": 1,
                  "at night": 1, "every morning": 1, "every night": 1, "every evening": 1, "twice a day": 2,
                  "twice per day": 2, "twice daily": 2, "bid": 2, "three times a day": 3, "three times per day": 3,
                  "three times daily": 3, "tid": 3, "four times a day": 4, "four times per day": 4,
                  "four times daily": 4, "qid": 4}

_YES = re.compile(r"^\W*(?:yes|yeah|yep|yup|i do|i have|i did|correct|right|definitely|a little|a bit|sometimes|"
                  r"kind of|sort of|uh[\s\-]?huh|mm[\s\-]?hmm)\b", _I)
_NO = re.compile(r"^\W*(?:no|nope|nah|not really|i don'?t|i do not|i haven'?t|i have not|none|never|"
                 r"not that i know(?: of)?|nothing)\b", _I)
_ASKS_ALLERGY = re.compile(r"\ballerg", _I)


@dataclass
class _Mention:
    type: EntityType
    normalized: str
    surface: str
    status: EntityStatus
    segment: Segment
    quote: str
    confidence: float
    value: str | None = None
    unit: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    explicit: bool = True
    extra_evidence: list[Evidence] = field(default_factory=list)


@dataclass
class _Question:
    segment: Segment
    sentence: str
    terms: list[TermMatch]
    about_allergy: bool


class RuleBasedExtractor:
    def __init__(self, matcher: LexiconMatcher | None = None):
        self.matcher = matcher or clinical_matcher()
        self._allergen_class_re = re.compile(
            r"\b(?:" + "|".join(sorted((re.escape(t) for t in kb.ALLERGEN_CLASS_TERMS), key=len, reverse=True)) + r")\b", _I)
        self._non_drug = {alias: canonical for canonical, aliases in kb.NON_DRUG_ALLERGENS.items() for alias in aliases}
        self._non_drug_re = re.compile(
            r"\b(?:" + "|".join(sorted((re.escape(a) for a in self._non_drug), key=len, reverse=True)) + r")\b", _I)
        reactions = {alias: canonical for canonical, aliases in terms.ALLERGY_REACTIONS.items() for alias in aliases}
        self._reactions = reactions
        self._reaction_re = re.compile(
            r"\b(?:" + "|".join(sorted((re.escape(a) for a in reactions), key=len, reverse=True)) + r")\b", _I)

    # ------------------------------------------------------------------ public

    def extract(self, transcript: Transcript) -> EntitySet:
        mentions: list[_Mention] = []
        last_question: _Question | None = None
        for segment in transcript.segments:
            seg_mentions, question = self._process_segment(segment)
            if segment.speaker == Speaker.PATIENT and last_question is not None:
                seg_mentions = self._resolve_answer(last_question, segment, seg_mentions)
            mentions.extend(seg_mentions)
            if segment.speaker in CLINICIAN_SPEAKERS or segment.speaker == Speaker.UNKNOWN:
                last_question = question
            else:
                last_question = None
        return aggregate(mentions)

    # ------------------------------------------------------------------ per segment

    def _process_segment(self, segment: Segment) -> tuple[list[_Mention], _Question | None]:
        mentions: list[_Mention] = []
        question: _Question | None = None
        sentences = split_sentences(segment.text)
        carry_consumed: dict[int, list[tuple[int, int]]] = {}
        last_with_entities: list[_Mention] = []

        for idx, sent in enumerate(sentences):
            if sent.is_question:
                question = _Question(segment, sent.text, self.matcher.find(sent.text), bool(_ASKS_ALLERGY.search(sent.text)))
                continue
            consumed = list(carry_consumed.get(idx, []))
            sent_mentions: list[_Mention] = []

            allergy_mentions, spans, next_spans = self._allergies(segment, sent, sentences[idx + 1] if idx + 1 < len(sentences) else None)
            sent_mentions += allergy_mentions
            consumed += spans
            if next_spans:
                carry_consumed.setdefault(idx + 1, []).extend(next_spans)

            sent_mentions += self._vitals(segment, sent)
            sent_mentions += self._demographics(segment, sent)
            sent_mentions += self._lexicon(segment, sent, consumed)

            # "It's getting worse." on its own applies to the previous sentence's symptoms.
            if not sent_mentions and last_with_entities:
                course = ctxlib.course_in(sent.text)
                if course:
                    for m in last_with_entities:
                        if m.type == EntityType.SYMPTOM and m.status == EntityStatus.PRESENT:
                            m.attributes.setdefault("course", course)
            if sent_mentions:
                last_with_entities = sent_mentions
            mentions += sent_mentions

        mentions += self._honorific(segment)
        return mentions, question

    # ------------------------------------------------------------------ allergies

    def _allergies(self, segment: Segment, sent: Sentence, next_sent: Sentence | None):
        text = sent.text
        mentions: list[_Mention] = []
        consumed: list[tuple[int, int]] = []
        next_consumed: list[tuple[int, int]] = []

        if _NO_ALLERGY.search(text):
            m = _NO_ALLERGY.search(text)
            mentions.append(self._mention(EntityType.ALLERGY, NKDA, m.group(), EntityStatus.ABSENT, segment, text, 0.9))
            consumed.append((m.start(), m.end()))

        spans: list[tuple[int, int, int]] = []  # (trigger_start, list_start, list_end)
        for trig in _ALLERGY_TRIGGER.finditer(text):
            if any(s <= trig.start() < e for s, e in consumed):
                continue
            end_match = _ALLERGY_LIST_END.search(text, trig.end())
            spans.append((trig.start(), trig.end(), end_match.start() if end_match else len(text)))
        for suf in _ALLERGY_SUFFIX.finditer(text):
            item_words = suf.group("item").lower().split()
            if item_words and item_words[-1] in _GENERIC_ALLERGY_WORDS:
                continue
            spans.append((suf.start(), suf.start(), suf.start("item") + len(suf.group("item"))))

        for trig_start, list_start, list_end in spans:
            found = self._allergens_in(text, list_start, list_end, allow_unknown=list_start != trig_start)
            if not found:
                continue
            ctx = ctxlib.analyze(text, trig_start, max(trig_start + 1, list_start))
            if ctx.family_relation:
                status = EntityStatus.FAMILY_MEMBER
            elif ctx.negated:
                status = EntityStatus.ABSENT
            elif ctx.uncertain:
                status = EntityStatus.POSSIBLE
            elif ctx.historical:
                status = EntityStatus.HISTORICAL
            else:
                status = EntityStatus.PRESENT
            consumed.append((trig_start, list_end))

            reaction, spans_ = self._reactions_in(text, list_end)
            consumed += spans_
            if reaction is None and next_sent is not None and _REACTION_BACKREF.search(next_sent.text):
                reaction, spans_ = self._reactions_in(next_sent.text, 0)
                next_consumed += spans_

            for normalized, surface, kind, cls in found:
                attrs = {"allergen_kind": kind}
                if cls:
                    attrs["allergen_class"] = cls
                if reaction:
                    attrs["reaction"] = reaction
                if ctx.family_relation:
                    attrs["relation"] = ctx.family_relation
                conf = 0.92 if kind != "unknown" else 0.6
                mentions.append(self._mention(EntityType.ALLERGY, normalized, surface, status, segment, text, conf, attrs=attrs))
        return mentions, consumed, next_consumed

    def _reactions_in(self, text: str, start: int) -> tuple[str | None, list[tuple[int, int]]]:
        """All reaction terms after `start` ("broke out in hives" -> hives), plus their spans to consume."""
        matches = list(self._reaction_re.finditer(text, start))
        if not matches:
            return None, []
        names = list(dict.fromkeys(self._reactions[m.group().lower()] for m in matches))
        if len(names) > 1 and any(m.group().lower() == "broke out" for m in matches):
            names = [n for n in names if n != "rash"] or names
        return ", ".join(names), [(m.start(), m.end()) for m in matches]

    def _allergens_in(self, text: str, start: int, end: int, allow_unknown: bool = True):
        """Return [(normalized, surface, kind, class_key)] for allergens inside text[start:end]."""
        span = text[start:end]
        found: list[tuple[str, str, str, str | None]] = []
        taken: list[tuple[int, int]] = []
        for m in self._allergen_class_re.finditer(span):
            cls = kb.ALLERGEN_CLASS_TERMS[m.group().lower()]
            found.append((_CLASS_ALLERGEN_NAMES.get(cls, cls), m.group(), "drug_class", cls))
            taken.append((m.start(), m.end()))
        for m in self.matcher.find(span, {EntityType.MEDICATION}):
            if any(s < m.end and m.start < e for s, e in taken):
                continue
            found.append((m.canonical, m.surface, "drug", None))
            taken.append((m.start, m.end))
        for m in self._non_drug_re.finditer(span):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            found.append((self._non_drug[m.group().lower()], m.group(), "non_drug", None))
            taken.append((m.start(), m.end()))
        if not found and allow_unknown:
            words = re.findall(r"[A-Za-z][A-Za-z\-]+", span)
            words = [w for w in words if w.lower() not in _GENERIC_ALLERGY_WORDS]
            if words and words[0].lower() not in {"anything", "something", "it", "them", "that", "everything", "nothing"}:
                found.append((words[0].lower(), words[0], "unknown", None))
        return found

    # ------------------------------------------------------------------ vitals / demographics

    def _vitals(self, segment: Segment, sent: Sentence) -> list[_Mention]:
        out: list[_Mention] = []
        for name, regex in _VITAL_PATTERNS:
            for m in regex.finditer(sent.text):
                value = re.sub(r"\s*(?:/|over)\s*", "/", m.group("v"))
                unit = _VITAL_UNITS.get(name)
                if name == "temperature":
                    raw_unit = (m.groupdict().get("u") or "").lower()
                    unit = "°C" if raw_unit.startswith("c") else "°F" if raw_unit.startswith("f") else (
                        "°F" if float(value) >= 50 else "°C")
                elif name == "weight":
                    unit = "kg" if m.group("u").lower().startswith("k") else "lb"
                out.append(self._mention(EntityType.VITAL, name, m.group(), EntityStatus.PRESENT, segment,
                                         sent.text, 0.9, value=value, unit=unit))
        return out

    def _demographics(self, segment: Segment, sent: Sentence) -> list[_Mention]:
        out: list[_Mention] = []
        text = sent.text
        for regex in _AGE_PATTERNS + ([_AGE_SELF] if segment.speaker == Speaker.PATIENT else []):
            for m in regex.finditer(text):
                if ctxlib.analyze(text, m.start(), m.end()).family_relation:
                    continue
                age = int(m.group("age"))
                if 0 < age <= 120:
                    out.append(self._mention(EntityType.DEMOGRAPHIC, "age", m.group(), EntityStatus.PRESENT, segment,
                                             text, 0.85, value=str(age), attrs={"basis": "stated"}))
        sex_patterns = _SEX_EXPLICIT + ([_SEX_SELF] if segment.speaker == Speaker.PATIENT else [])
        for regex in sex_patterns:
            for m in regex.finditer(text):
                if ctxlib.analyze(text, m.start(), m.end()).family_relation:
                    continue
                out.append(self._mention(EntityType.DEMOGRAPHIC, "sex", m.group(), EntityStatus.PRESENT, segment,
                                         text, 0.9, value=_SEX_MAP[m.group("sex").lower()], attrs={"basis": "stated"}))
        return _dedupe_same_value(out)

    def _honorific(self, segment: Segment) -> list[_Mention]:
        if segment.speaker not in CLINICIAN_SPEAKERS:
            return []
        out = []
        for m in _HONORIFIC.finditer(segment.text):
            out.append(self._mention(EntityType.DEMOGRAPHIC, "sex", m.group("h") + ".", EntityStatus.PRESENT, segment,
                                     segment.text, 0.7, value=_SEX_MAP[m.group("h").lower()],
                                     attrs={"basis": "form_of_address"}))
        return out[:1]

    # ------------------------------------------------------------------ lexicon terms

    def _lexicon(self, segment: Segment, sent: Sentence, consumed: list[tuple[int, int]]) -> list[_Mention]:
        text = sent.text
        matches = [m for m in self.matcher.find(text) if not any(s < m.end and m.start < e for s, e in consumed)]
        med_starts = [m.start for m in matches if m.type == EntityType.MEDICATION]
        duration = ctxlib.duration_in(text)
        course = ctxlib.course_in(text)
        out: list[_Mention] = []

        for m in matches:
            if m.type == EntityType.DIAGNOSIS and re.match(r"[\s\-]?like\b", text[m.end:], _I):
                continue  # "flu-like symptoms" is not influenza
            if m.type == EntityType.SYMPTOM and _REACTION_DESCRIPTION.search(text[:m.start]):
                continue  # "they give me a rash" describes a drug reaction, not a current symptom
            ctx = ctxlib.analyze(text, m.start, m.end)
            attrs: dict[str, str] = {}

            if m.type == EntityType.MEDICATION:
                if _DRUG_REACTION_POST.match(text[m.end:]) and self._reaction_re.search(text, m.end, m.end + 50):
                    reaction = self._reactions[self._reaction_re.search(text, m.end, m.end + 50).group().lower()]
                    out.append(self._mention(EntityType.ALLERGY, m.canonical, m.surface, EntityStatus.PRESENT, segment,
                                             text, 0.85, attrs={"allergen_kind": "drug", "reaction": reaction}))
                    continue
                next_med = min((s for s in med_starts if s > m.start), default=len(text))
                attrs = _dosing(text, m.start, m.end, next_med)
                status = ctxlib.resolve_status(m.type, ctx, segment.speaker, has_dosing=bool(attrs))
                if ctx.med_cue in ("AVOID", "AVOID_NEAR"):
                    attrs["intent"] = "avoid"
                conf = 0.9 if ctx.med_cue else 0.7
            else:
                status = ctxlib.resolve_status(m.type, ctx, segment.speaker)
                conf = 0.9 if not ctx.uncertain else 0.75
                if m.type == EntityType.SYMPTOM:
                    if ctx.severity:
                        attrs["severity"] = ctx.severity
                    if duration and status in (EntityStatus.PRESENT, EntityStatus.POSSIBLE):
                        attrs["duration"] = duration
                    if course and status == EntityStatus.PRESENT:
                        attrs["course"] = course
                if m.type == EntityType.PROCEDURE and ctx.result:
                    attrs["result"] = ctx.result
                if m.type == EntityType.DIAGNOSIS:
                    attrs["stated_by"] = segment.speaker.value
                    weeks = re.search(r"(\d{1,2})\s+weeks?\s+pregnant", text, _I)
                    if m.canonical == "pregnancy" and weeks:
                        attrs["gestational_weeks"] = weeks.group(1)
            if ctx.family_relation:
                attrs["relation"] = ctx.family_relation
            out.append(self._mention(m.type, m.canonical, m.surface, status, segment, text, conf, attrs=attrs))
        return out

    # ------------------------------------------------------------------ dialogue inference

    def _resolve_answer(self, question: _Question, answer: Segment, mentions: list[_Mention]) -> list[_Mention]:
        first = split_sentences(answer.text)
        first_text = first[0].text if first else answer.text
        yes, no = bool(_YES.match(first_text)), bool(_NO.match(first_text))
        q_evidence = Evidence.from_segment(question.segment, question.sentence)

        if question.about_allergy:
            has_allergy = any(m.type == EntityType.ALLERGY for m in mentions)
            if not has_allergy:
                # "Any allergies?" -> "Penicillin." : medication mentions in the answer are allergens.
                converted = []
                for m in mentions:
                    if m.type == EntityType.MEDICATION and m.status in (EntityStatus.CURRENT, EntityStatus.UNKNOWN):
                        converted.append(_Mention(EntityType.ALLERGY, m.normalized, m.surface, EntityStatus.PRESENT,
                                                  answer, m.quote, 0.8, attributes={"allergen_kind": "drug",
                                                                                    "inferred_from_dialogue": "true"},
                                                  extra_evidence=[q_evidence]))
                    else:
                        converted.append(m)
                mentions = converted
                if not any(m.type == EntityType.ALLERGY for m in mentions):
                    reaction, _ = self._reactions_in(first_text, 0)
                    for normalized, surface, kind, cls in self._allergens_in(first_text, 0, len(first_text), allow_unknown=False):
                        if kind == "non_drug" or kind == "drug_class":
                            attrs = {"allergen_kind": kind, "inferred_from_dialogue": "true"}
                            if cls:
                                attrs["allergen_class"] = cls
                            if reaction:
                                attrs["reaction"] = reaction
                            mentions.append(_Mention(EntityType.ALLERGY, normalized, surface, EntityStatus.PRESENT,
                                                     answer, first_text, 0.75, attributes=attrs, extra_evidence=[q_evidence]))
                if no and not any(m.type == EntityType.ALLERGY for m in mentions):
                    mentions.append(_Mention(EntityType.ALLERGY, NKDA, first_text, EntityStatus.ABSENT, answer,
                                             first_text, 0.75, attributes={"inferred_from_dialogue": "true"},
                                             explicit=False, extra_evidence=[q_evidence]))

        answered = {m.normalized for m in mentions}
        q_terms = [t for t in question.terms if t.type in (EntityType.SYMPTOM, EntityType.DIAGNOSIS, EntityType.MEDICATION)
                   and t.canonical not in answered]
        if not q_terms or not (yes or no):
            return mentions
        same_type_answered = any(m.type in {t.type for t in q_terms} for m in mentions)

        for term in q_terms:
            if no:
                status, conf = EntityStatus.ABSENT, 0.7
            elif same_type_answered:
                continue  # patient answered "yes, X" — do not assume the other options
            elif len(q_terms) == 1:
                status = EntityStatus.CURRENT if term.type == EntityType.MEDICATION else EntityStatus.PRESENT
                conf = 0.75
            else:
                status, conf = EntityStatus.POSSIBLE, 0.5  # "Any cough or fever?" "Yes." is ambiguous
            mentions.append(_Mention(term.type, term.canonical, term.surface, status, answer, first_text, conf,
                                     attributes={"inferred_from_dialogue": "true"}, explicit=False,
                                     extra_evidence=[q_evidence]))
        return mentions

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _mention(etype, normalized, surface, status, segment, quote, conf, value=None, unit=None, attrs=None) -> _Mention:
        asr = segment.confidence if segment.confidence is not None else 1.0
        return _Mention(etype, normalized, surface, status, segment, quote.strip(), round(conf * asr, 3),
                        value=value, unit=unit, attributes=dict(attrs or {}))


_CLASS_ALLERGEN_NAMES = {
    "sulfonamide_antibiotics": "sulfa drugs",
    "nsaids": "nsaids",
    "cephalosporins": "cephalosporins",
    "macrolides": "macrolides",
    "opioids": "opioids",
    "fluoroquinolones": "fluoroquinolones",
    "tetracyclines": "tetracyclines",
}


def _dedupe_same_value(mentions: list[_Mention]) -> list[_Mention]:
    seen, out = set(), []
    for m in mentions:
        key = (m.normalized, m.value)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _dosing(text: str, start: int, end: int, window_end: int) -> dict[str, str]:
    after = text[end:window_end]
    attrs: dict[str, str] = {}
    dose = _DOSE.search(after) or _DOSE_BEFORE.search(text[max(0, start - 25):start])
    if dose:
        unit = dose.group("unit").lower()
        unit = {"milligram": "mg", "milligrams": "mg", "microgram": "mcg", "micrograms": "mcg", "gram": "g",
                "grams": "g", "milliliter": "ml", "milliliters": "ml"}.get(unit, unit)
        attrs["dose"] = f"{dose.group('dose')} {unit}"
        amount = float(dose.group("dose"))
        if unit in ("mg", "g", "mcg"):
            attrs["dose_mg"] = f"{amount * {'mg': 1, 'g': 1000, 'mcg': 0.001}[unit]:g}"
    freq = _FREQUENCY.search(after)
    if freq:
        attrs["frequency"] = freq.group(1).lower()
        per_day = _DOSES_PER_DAY.get(attrs["frequency"])
        every = re.match(r"(?:every (\d+) hours|q(\d+)h)", attrs["frequency"])
        if every:
            per_day = 24 // int(every.group(1) or every.group(2))
        if per_day:
            attrs["doses_per_day"] = str(per_day)
    dur = _MED_DURATION.search(after)
    if dur:
        attrs["duration"] = f"{dur.group('n')} {dur.group('u')}".lower()
    route = _ROUTE.search(after)
    if route:
        attrs["route"] = route.group(1).lower()
    return attrs


# --------------------------------------------------------------------------- aggregation

_KEY_WITH_VALUE = {EntityType.VITAL, EntityType.DEMOGRAPHIC}


def aggregate(mentions: list[_Mention]) -> EntitySet:
    groups: dict[tuple, list[_Mention]] = {}
    for m in mentions:
        key = (m.type, m.normalized, m.value if m.type in _KEY_WITH_VALUE else None)
        groups.setdefault(key, []).append(m)

    entities: list[ClinicalEntity] = []
    for (etype, normalized, _), group in groups.items():
        group.sort(key=lambda m: m.segment.start_time)
        explicit = [m for m in group if m.explicit]
        deciding = explicit or group
        status = deciding[-1].status
        if etype == EntityType.ALLERGY:
            # Fail safe: any affirmed allergy keeps the allergy active (inconsistencies are flagged by a rule).
            active = [m for m in deciding if m.status in ACTIVE_ALLERGY_STATUSES]
            if active:
                status = active[-1].status

        attributes: dict[str, str] = {}
        for m in group:
            for k, v in m.attributes.items():
                attributes.setdefault(k, v)
        if explicit and all("inferred_from_dialogue" not in m.attributes for m in explicit):
            attributes.pop("inferred_from_dialogue", None)

        evidence: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        for m in group:
            for ev in [*m.extra_evidence, Evidence.from_segment(m.segment, m.quote)]:
                if (ev.segment_id, ev.quote) not in seen:
                    seen.add((ev.segment_id, ev.quote))
                    evidence.append(ev)
        evidence.sort(key=lambda e: e.start_time)

        entities.append(ClinicalEntity(
            type=etype,
            text=group[0].surface,
            normalized=normalized,
            status=status,
            value=group[0].value,
            unit=group[0].unit,
            attributes=attributes,
            evidence=evidence,
            confidence=max(m.confidence for m in deciding),
            source="rule",
            status_history=[m.status for m in explicit],
        ))
    return EntitySet(items=assign_ids(entities))


def assign_ids(entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
    # Stable sort: entities from the same segment keep the order in which they were spoken.
    entities = sorted(entities, key=lambda e: e.first_time)
    for i, entity in enumerate(entities, start=1):
        entity.entity_id = f"E{i:03d}"
    return entities
