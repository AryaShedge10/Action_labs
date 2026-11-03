"""NegEx-style context analysis for a term mention inside a sentence.

For every mention we look at the *clause* that contains it (clauses are delimited by
"but", "however", "just", ";", ...) and classify negation, family attribution, temporality,
uncertainty and — for medications — the prescribing intent. When several medication cues
appear before a mention, the one closest to the mention wins
("avoid amoxicillin and use azithromycin" -> amoxicillin avoided, azithromycin prescribed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import CLINICIAN_SPEAKERS, EntityStatus, EntityType, Speaker

_I = re.IGNORECASE

_CLAUSE_BREAK = re.compile(
    r"\b(?:but|however|although|though|except|apart from|aside from|other than|just|only|still|whereas|while)\b"
    r"|;|\s[-–—]\s|—"
    r"|\band\s+(?:now\s+)?(?:i|my|he|she|it|we|they|there|you)\b"
    r"|,\s*(?:so|because|which|then)\b"
    r"|(?:^|[\s,])(?:no|nope|yes|yeah|yep|okay|ok|well)\s*,",
    _I,
)

# Phrases that contain a negation word but do not negate the following term.
_PSEUDO_NEGATION = re.compile(
    r"\b(?:not sure|not certain|no longer|not only|not just|no change in|no improvement in|not getting better|"
    r"not improving|doesn'?t help|didn'?t help|not helping|not working|no relief|gram[\s\-]negative|"
    r"no doubt|not really sure)\b",
    _I,
)

_NEGATION_PRE = re.compile(
    r"\b(?:no|not|denies|denied|deny|denying|without|never|negative for|free of|absence of|none|nor|neither|"
    r"haven'?t (?:had|been|got)|have not (?:had|been)|hasn'?t|didn'?t have|did not have|don'?t have|do not have|"
    r"doesn'?t have|does not have|isn'?t|aren'?t|wasn'?t|no signs? of|no evidence of|ruled out|rules out|"
    r"don'?t think (?:it'?s|i have|you have)|doesn'?t look like|does not look like|not consistent with|"
    r"don'?t take|do not take|doesn'?t take|not taking|stopped having)\b",
    _I,
)
_NEGATION_POST = re.compile(
    r"^\W*(?:\w+\W+){0,3}?(?:was |is |has been |been )?(?:ruled out|unlikely|negative)\b", _I)

_RESOLVED_POST = re.compile(r"\b(?:went away|is gone|has gone|gone now|resolved|cleared up|better now|has stopped)\b", _I)
_RESOLVED_FOLLOWING = re.compile(
    r"\bbut\s+(?:it'?s|it is|it has|they'?re|they are|that'?s)\s+(?:all\s+)?(?:gone|resolved|better|cleared|over|fine)\b", _I)

_FAMILY = re.compile(
    r"\b(?:my|his|her|our|your)\s+(?P<rel>mother|mom|mum|father|dad|brother|sister|son|daughter|wife|husband|"
    r"grandmother|grandfather|grandma|grandpa|aunt|uncle|cousin|child|kid|kids|children|baby|parents|"
    r"family|partner|roommate|friend|coworker)\b|\b(?P<fh>family history)\b|\bruns in (?:my|the) family\b",
    _I,
)

_HISTORICAL = re.compile(
    r"\b(?:used to|in the past|last year|years ago|months ago|as a child|when i was (?:a kid|young|little)|"
    r"history of|previously|formerly|back in \d{4}|a while back|long time ago|had .{0,30} before)\b",
    _I,
)

_UNCERTAIN = re.compile(
    r"\b(?:maybe|might|might be|may have|may be|possibly|possible|probably|probable|likely|i think|i guess|"
    r"not sure|could be|suspect|suspected|concern for|concerned about|worried about|looks like|seems like|"
    r"sounds like|rule out|r/o|consistent with|suggestive of|differential)\b",
    _I,
)

# Medication intent cues: (kind, regex). The cue that ends closest to the mention wins.
_MED_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("NEGATED", re.compile(r"\b(?:no|not|never|don'?t take|do not take|doesn'?t take|not taking|haven'?t taken|"
                           r"not on|allergic|allergy)\b", _I)),
    ("AVOID", re.compile(r"\b(?:avoid|avoiding|steer clear of|stay away from|won'?t (?:give|prescribe|use|start)|"
                         r"will not (?:give|prescribe|use|start)|not (?:going to )?(?:give|prescribe|start|use)|"
                         r"hold(?:ing)? off (?:on)?|can'?t take|cannot take|can not take|shouldn'?t take|"
                         r"should not take|no more|not safe|contraindicated)\b", _I)),
    ("AVOID_NEAR", re.compile(r"\b(?:instead of|rather than|in place of)\s*$", _I)),
    ("STOPPED", re.compile(r"\b(?:stopped|stop taking|quit|discontinued?|no longer (?:take|taking|on)|came off|"
                           r"went off|ran out of)\b", _I)),
    ("HISTORICAL", re.compile(r"\b(?:used to (?:take|be on)|was on|were on|previously (?:took|on)|in the past|"
                              r"last year|years ago|tried .{0,20} before)\b", _I)),
    ("PRESCRIBED", re.compile(r"\b(?:prescrib\w*|start (?:you )?on|starting you on|put (?:you )?on|putting you on|"
                              r"i'?ll give you|i will give you|i'?m giving you|going to give you|we'?ll (?:start|give|use|try)|"
                              r"let'?s (?:start|try|use|give|go with)|switch (?:you )?to|send (?:in )?a script|"
                              r"script for|go ahead with|we will use|i'?d like (?:you|to) (?:to )?start|order(?:ing)?|"
                              r"(?:i'?ll|i will|we'?ll|we will|let'?s) (?:also )?add)\b", _I)),
    ("RECOMMENDED", re.compile(r"\b(?:recommend\w*|suggest\w*|you (?:can|could|may|should) (?:take|use|try)|"
                               r"(?:we|i) (?:can|could) (?:give|use|try)|give (?:him|her|them|the child)|"
                               r"over the counter|otc|try (?:some|taking)|feel free to (?:take|use)|"
                               r"told me to take|said to take|advised me to take)\b", _I)),
    ("CURRENT", re.compile(r"\b(?:i take|i'?m taking|i am taking|i'?ve been taking|i have been taking|i'?m on|i am on|"
                           r"currently (?:on|taking)|been on|i use|i'?ve been using|i took|i have taken|i'?ve taken|"
                           r"i tried|continue|keep taking|still taking|my medications? (?:is|are|include))\b", _I)),
    ("TAKE", re.compile(r"\b(?:take|taking|use)\b", _I)),
]

_NEGATIVE_MED_CUES = {"NEGATED", "AVOID", "AVOID_NEAR", "STOPPED"}
_CONJUNCTION = re.compile(r"\b(?:and|but|then|so|instead)\b", _I)

_PROCEDURE_ORDER = re.compile(
    r"\b(?:let'?s (?:do|get|run|check|order|take|send)|i'?ll (?:order|do|get|run|send|check)|i will (?:order|do|get|run)|"
    r"we'?ll (?:do|get|run|order|check|send)|going to (?:do|order|get|run|check|examine|send)|i want (?:to|you to)|"
    r"need (?:a|an|to)|order(?:ing)?|schedule|refer\w*|book|recommend\w*|should (?:get|have))\b", _I)
_PROCEDURE_DONE = re.compile(r"\b(?:came back|showed|shows|results?|was positive|was negative|performed|did a|did the)\b", _I)
_RESULT = re.compile(r"\b(positive|negative|normal|abnormal|unremarkable|elevated|low)\b", _I)

_SEVERITY = re.compile(
    r"\b(severe|severely|mild|mildly|moderate|low[\s\-]grade|high[\s\-]grade|slight|slightly|bad|terrible|"
    r"awful|excruciating|intense|constant|intermittent)\s+(?:\w+\s+)?$", _I)
_DURATION = re.compile(
    r"\b(?:for|over|since|x)\s+(?:about\s+|around\s+|almost\s+|nearly\s+)?(?:the\s+)?(?:past\s+|last\s+)?"
    r"(?P<dur>(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a few|several|a|an)\s+"
    r"(?:hours?|days?|weeks?|months?|years?)|yesterday|last night|this morning|a while)\b"
    r"|\b(?P<ago>(?:\d+|one|two|three|four|five|six|seven|a couple of|a few|a)\s+(?:hours?|days?|weeks?|months?)\s+ago)\b",
    _I,
)
_COURSE = re.compile(r"\b(getting worse|worsening|worse|getting better|improving|better)\b", _I)


@dataclass
class MentionContext:
    negated: bool = False
    family_relation: str | None = None
    historical: bool = False
    resolved: bool = False
    uncertain: bool = False
    med_cue: str | None = None
    procedure_ordered: bool = False
    procedure_done: bool = False
    result: str | None = None
    severity: str | None = None


def _mask(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), text)


def clause_bounds(sentence: str, start: int, end: int) -> tuple[int, int]:
    clause_start, clause_end = 0, len(sentence)
    for m in _CLAUSE_BREAK.finditer(sentence):
        if m.end() <= start:
            clause_start = max(clause_start, m.end())
        elif m.start() >= end:
            clause_end = min(clause_end, m.start())
            break
    return clause_start, clause_end


def analyze(sentence: str, start: int, end: int) -> MentionContext:
    """Analyse a mention spanning sentence[start:end]."""
    c_start, c_end = clause_bounds(sentence, start, end)
    pre = _mask(sentence[c_start:start], _PSEUDO_NEGATION)
    post = _mask(sentence[end:c_end], _PSEUDO_NEGATION)
    clause = sentence[c_start:c_end]
    ctx = MentionContext()

    ctx.negated = bool(_NEGATION_PRE.search(pre) or _NEGATION_POST.search(post))
    fam = _FAMILY.search(clause)
    if fam:
        ctx.family_relation = (fam.group("rel") or "family").lower()
    ctx.historical = bool(_HISTORICAL.search(clause))
    ctx.resolved = bool(_RESOLVED_POST.search(post) or _RESOLVED_FOLLOWING.search(sentence[end:]))
    ctx.uncertain = bool(_UNCERTAIN.search(sentence[c_start:start]))  # unmasked: "not sure" is an uncertainty cue

    best: tuple[int, str] | None = None
    raw_pre = sentence[c_start:start]
    for kind, regex in _MED_CUES:
        for m in regex.finditer(raw_pre):
            if kind == "TAKE" and best is not None:
                # Generic "take/use" only overrides a negative cue across a conjunction:
                # "avoid amoxicillin and use azithromycin" (override) vs "avoid taking ibuprofen" (no override).
                negative_before = best[1] in _NEGATIVE_MED_CUES and m.start() >= best[0]
                if not (negative_before and _CONJUNCTION.search(raw_pre, best[0], m.start())):
                    continue
            if best is None or m.end() > best[0] or (m.end() == best[0] and best[1] == "TAKE"):
                best = (m.end(), kind)
    ctx.med_cue = best[1] if best else None
    if ctx.med_cue is None and re.search(r"\bover the counter\b|\botc\b", post, _I):
        ctx.med_cue = "RECOMMENDED"
    if re.match(r"^\W*(?:was |were )?(?:stopped|discontinued)\b", post, _I):
        ctx.med_cue = "STOPPED"

    ctx.procedure_ordered = bool(_PROCEDURE_ORDER.search(pre))
    ctx.procedure_done = bool(_PROCEDURE_DONE.search(clause))
    result = _RESULT.search(sentence[end:]) or _RESULT.search(pre)
    ctx.result = result.group(1).lower() if result else None

    sev = _SEVERITY.search(sentence[max(0, start - 30):start])
    if sev:
        ctx.severity = re.sub(r"ly$", "", sev.group(1).lower()).replace(" ", "-").replace("slight", "mild")
    return ctx


def duration_in(text: str) -> str | None:
    m = _DURATION.search(text)
    if not m:
        return None
    return (m.group("dur") or m.group("ago")).lower()


def course_in(text: str) -> str | None:
    m = _COURSE.search(text)
    if not m:
        return None
    word = m.group(1).lower()
    return "worsening" if "wors" in word else "improving"


def resolve_status(etype: EntityType, ctx: MentionContext, speaker: Speaker, has_dosing: bool = False) -> EntityStatus:
    """Map a mention context to an entity status for the given entity type."""
    if ctx.family_relation:
        return EntityStatus.FAMILY_MEMBER
    clinician = speaker in CLINICIAN_SPEAKERS

    if etype == EntityType.MEDICATION:
        cue = ctx.med_cue
        if cue in ("NEGATED", "AVOID", "AVOID_NEAR"):
            return EntityStatus.ABSENT
        if cue == "STOPPED":
            return EntityStatus.STOPPED
        if cue == "HISTORICAL":
            return EntityStatus.HISTORICAL
        if cue == "PRESCRIBED":
            return EntityStatus.PRESCRIBED
        if cue == "RECOMMENDED":
            return EntityStatus.RECOMMENDED
        if cue == "CURRENT":
            return EntityStatus.CURRENT
        if cue == "TAKE":
            return EntityStatus.PRESCRIBED if clinician else EntityStatus.CURRENT
        if ctx.negated:
            return EntityStatus.ABSENT
        if clinician:
            return EntityStatus.PRESCRIBED if has_dosing else EntityStatus.UNKNOWN
        return EntityStatus.CURRENT if speaker == Speaker.PATIENT else EntityStatus.UNKNOWN

    if etype == EntityType.PROCEDURE:
        if ctx.negated and not ctx.procedure_done:
            return EntityStatus.ABSENT
        if ctx.historical:
            return EntityStatus.HISTORICAL
        if ctx.procedure_done or ctx.result:
            return EntityStatus.PRESENT
        if ctx.procedure_ordered or clinician:
            return EntityStatus.RECOMMENDED
        return EntityStatus.PRESENT

    # SYMPTOM / FINDING / DIAGNOSIS
    if ctx.negated:
        return EntityStatus.ABSENT
    if ctx.resolved or ctx.historical:
        return EntityStatus.HISTORICAL
    if ctx.uncertain:
        return EntityStatus.POSSIBLE
    return EntityStatus.PRESENT
