"""Drug knowledge base used by extraction (lexicon) and the safety rules.

Scope: common primary-care / urgent-care drugs. This is a curated teaching dataset, NOT a
substitute for RxNorm / a commercial interaction database. Every entry should be reviewed by
a clinician before the rule version is bumped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Drug:
    name: str                                   # canonical generic name (lower-case)
    classes: frozenset[str]
    aliases: tuple[str, ...] = ()               # brand names / spoken variants
    max_single_dose_mg: float | None = None     # adult, oral, conservative upper bound
    max_daily_dose_mg: float | None = None


def _d(name: str, classes: set[str], aliases: tuple[str, ...] = (), single: float | None = None,
       daily: float | None = None) -> Drug:
    return Drug(name, frozenset(classes), aliases, single, daily)


_DRUG_LIST: list[Drug] = [
    # --- beta-lactams
    _d("amoxicillin", {"penicillins", "beta_lactams", "antibiotics"}, ("amoxil", "amoxycillin"), 1000, 4000),
    _d("amoxicillin-clavulanate", {"penicillins", "beta_lactams", "antibiotics"},
       ("augmentin", "amoxicillin clavulanate", "amox-clav", "co-amoxiclav"), 1000, 4000),
    _d("ampicillin", {"penicillins", "beta_lactams", "antibiotics"}),
    _d("penicillin", {"penicillins", "beta_lactams", "antibiotics"},
       ("penicillin v", "penicillin vk", "pen vk", "penicillin g", "penicillins")),
    _d("dicloxacillin", {"penicillins", "beta_lactams", "antibiotics"}),
    _d("piperacillin-tazobactam", {"penicillins", "beta_lactams", "antibiotics"}, ("zosyn",)),
    _d("cephalexin", {"cephalosporins", "beta_lactams", "antibiotics"}, ("keflex",), 1000, 4000),
    _d("cefuroxime", {"cephalosporins", "beta_lactams", "antibiotics"}, ("ceftin",)),
    _d("cefdinir", {"cephalosporins", "beta_lactams", "antibiotics"}, ("omnicef",)),
    _d("ceftriaxone", {"cephalosporins", "beta_lactams", "antibiotics"}, ("rocephin",)),
    _d("cefazolin", {"cephalosporins", "beta_lactams", "antibiotics"}),
    _d("meropenem", {"carbapenems", "beta_lactams", "antibiotics"}),
    _d("imipenem", {"carbapenems", "beta_lactams", "antibiotics"}),
    # --- other antibiotics / anti-infectives
    _d("azithromycin", {"macrolides", "antibiotics"}, ("zithromax", "z-pak", "z pak", "zpack", "z-pack"), 2000),
    _d("clarithromycin", {"macrolides", "antibiotics", "strong_cyp3a4_inhibitors"}, ("biaxin",)),
    _d("erythromycin", {"macrolides", "antibiotics", "strong_cyp3a4_inhibitors"}),
    _d("doxycycline", {"tetracyclines", "antibiotics"}, ("vibramycin", "doryx"), 200, 200),
    _d("tetracycline", {"tetracyclines", "antibiotics"}),
    _d("minocycline", {"tetracyclines", "antibiotics"}),
    _d("ciprofloxacin", {"fluoroquinolones", "antibiotics"}, ("cipro",), 750, 1500),
    _d("levofloxacin", {"fluoroquinolones", "antibiotics"}, ("levaquin",), 750, 750),
    _d("sulfamethoxazole-trimethoprim", {"sulfonamide_antibiotics", "antibiotics"},
       ("bactrim", "septra", "tmp-smx", "co-trimoxazole", "sulfamethoxazole", "trimethoprim sulfamethoxazole"), 1600),
    _d("nitrofurantoin", {"antibiotics"}, ("macrobid", "macrodantin")),
    _d("metronidazole", {"antibiotics"}, ("flagyl",)),
    _d("clindamycin", {"antibiotics"}, ("cleocin",)),
    _d("fluconazole", {"antifungals", "cyp_inhibitors"}, ("diflucan",)),
    _d("oseltamivir", {"antivirals"}, ("tamiflu",)),
    # --- analgesics / NSAIDs
    _d("ibuprofen", {"nsaids"}, ("advil", "motrin"), 800, 3200),
    _d("naproxen", {"nsaids"}, ("aleve", "naprosyn"), 550, 1500),
    _d("aspirin", {"nsaids", "salicylates", "antiplatelets"}, ("acetylsalicylic acid",), 1000, 4000),
    _d("diclofenac", {"nsaids"}, ("voltaren",), 75, 150),
    _d("celecoxib", {"nsaids"}, ("celebrex",), 400, 400),
    _d("meloxicam", {"nsaids"}, ("mobic",), 15, 15),
    _d("ketorolac", {"nsaids"}, ("toradol",)),
    _d("indomethacin", {"nsaids"}),
    _d("acetaminophen", {"analgesics"}, ("tylenol", "paracetamol", "panadol"), 1000, 4000),
    _d("tramadol", {"opioids", "serotonergic"}, ("ultram",), 100, 400),
    _d("codeine", {"opioids"}),
    _d("oxycodone", {"opioids"}, ("oxycontin", "percocet")),
    _d("hydrocodone", {"opioids"}, ("vicodin", "norco")),
    _d("morphine", {"opioids"}),
    # --- cardiovascular
    _d("warfarin", {"anticoagulants", "vitamin_k_antagonists"}, ("coumadin", "jantoven"), 20),
    _d("apixaban", {"anticoagulants"}, ("eliquis",)),
    _d("rivaroxaban", {"anticoagulants"}, ("xarelto",)),
    _d("heparin", {"anticoagulants"}),
    _d("clopidogrel", {"antiplatelets"}, ("plavix",)),
    _d("lisinopril", {"ace_inhibitors"}, ("zestril", "prinivil"), 80, 80),
    _d("enalapril", {"ace_inhibitors"}, ("vasotec",)),
    _d("ramipril", {"ace_inhibitors"}, ("altace",)),
    _d("losartan", {"arbs"}, ("cozaar",), 100, 100),
    _d("valsartan", {"arbs"}, ("diovan",)),
    _d("amlodipine", {"calcium_channel_blockers"}, ("norvasc",), 10, 10),
    _d("metoprolol", {"beta_blockers"}, ("lopressor", "toprol")),
    _d("atenolol", {"beta_blockers"}, ("tenormin",)),
    _d("hydrochlorothiazide", {"thiazide_diuretics"}, ("hctz",)),
    _d("furosemide", {"loop_diuretics"}, ("lasix",)),
    _d("spironolactone", {"potassium_sparing_diuretics"}, ("aldactone",)),
    _d("atorvastatin", {"statins"}, ("lipitor",), 80, 80),
    _d("simvastatin", {"statins"}, ("zocor",), 80, 80),
    _d("rosuvastatin", {"statins"}, ("crestor",), 40, 40),
    _d("nitroglycerin", {"nitrates"}, ("nitrostat", "nitro patch", "glyceryl trinitrate")),
    _d("isosorbide mononitrate", {"nitrates"}, ("imdur", "isosorbide")),
    _d("digoxin", {"cardiac_glycosides"}, ("lanoxin",)),
    _d("amiodarone", {"antiarrhythmics", "cyp_inhibitors"}, ("cordarone",)),
    _d("sildenafil", {"pde5_inhibitors"}, ("viagra", "revatio"), 100),
    _d("tadalafil", {"pde5_inhibitors"}, ("cialis",), 20),
    # --- endocrine / metabolic
    _d("metformin", {"biguanides"}, ("glucophage",), 2000, 2550),
    _d("insulin", {"insulins"}, ("lantus", "humalog", "novolog")),
    _d("glipizide", {"sulfonylureas"}, ("glucotrol",)),
    _d("levothyroxine", {"thyroid_hormones"}, ("synthroid", "levoxyl")),
    _d("prednisone", {"corticosteroids"}, (), 100),
    _d("methylprednisolone", {"corticosteroids"}, ("medrol",)),
    # --- CNS
    _d("sertraline", {"ssris", "serotonergic"}, ("zoloft",), 200, 200),
    _d("fluoxetine", {"ssris", "serotonergic"}, ("prozac",), 80, 80),
    _d("citalopram", {"ssris", "serotonergic"}, ("celexa",), 40, 40),
    _d("escitalopram", {"ssris", "serotonergic"}, ("lexapro",), 20, 20),
    _d("paroxetine", {"ssris", "serotonergic"}, ("paxil",), 60, 60),
    _d("phenelzine", {"maois", "serotonergic"}, ("nardil",)),
    _d("selegiline", {"maois"}, ("emsam",)),
    _d("tranylcypromine", {"maois", "serotonergic"}, ("parnate",)),
    _d("alprazolam", {"benzodiazepines"}, ("xanax",)),
    _d("lorazepam", {"benzodiazepines"}, ("ativan",)),
    _d("diazepam", {"benzodiazepines"}, ("valium",)),
    _d("lithium", {"mood_stabilizers"}),
    _d("valproate", {"anticonvulsants", "teratogens"}, ("valproic acid", "depakote", "divalproex")),
    _d("tizanidine", {"muscle_relaxants"}, ("zanaflex",)),
    # --- respiratory / allergy / GI
    _d("albuterol", {"short_acting_beta_agonists"}, ("ventolin", "proair", "salbutamol")),
    _d("fluticasone", {"corticosteroids"}, ("flonase", "flovent")),
    _d("montelukast", {"leukotriene_antagonists"}, ("singulair",)),
    _d("cetirizine", {"antihistamines"}, ("zyrtec",)),
    _d("loratadine", {"antihistamines"}, ("claritin",)),
    _d("diphenhydramine", {"antihistamines", "first_gen_antihistamines"}, ("benadryl",)),
    _d("omeprazole", {"ppis", "cyp2c19_inhibitors"}, ("prilosec",)),
    _d("pantoprazole", {"ppis"}, ("protonix",)),
    _d("ondansetron", {"antiemetics"}, ("zofran",)),
    # --- other high-risk
    _d("methotrexate", {"antimetabolites", "teratogens"}),
    _d("isotretinoin", {"retinoids", "teratogens"}, ("accutane",)),
    _d("misoprostol", {"prostaglandins", "teratogens"}, ("cytotec",)),
]

DRUGS: dict[str, Drug] = {d.name: d for d in _DRUG_LIST}

CLASS_LABELS: dict[str, str] = {
    "penicillins": "penicillin-class antibiotics",
    "cephalosporins": "cephalosporins",
    "carbapenems": "carbapenems",
    "macrolides": "macrolide antibiotics",
    "tetracyclines": "tetracyclines",
    "fluoroquinolones": "fluoroquinolones",
    "sulfonamide_antibiotics": "sulfonamide (sulfa) antibiotics",
    "nsaids": "NSAIDs",
    "opioids": "opioids",
    "anticoagulants": "anticoagulants",
    "antiplatelets": "antiplatelet agents",
    "ace_inhibitors": "ACE inhibitors",
    "arbs": "angiotensin receptor blockers",
    "statins": "statins",
    "ssris": "SSRIs",
    "maois": "MAO inhibitors",
    "benzodiazepines": "benzodiazepines",
    "nitrates": "nitrates",
    "pde5_inhibitors": "PDE5 inhibitors",
    "potassium_sparing_diuretics": "potassium-sparing diuretics",
    "thiazide_diuretics": "thiazide diuretics",
    "first_gen_antihistamines": "first-generation antihistamines",
}

# A drug allergy only generalises along these classes. Broad buckets such as "antibiotics"
# are deliberately excluded (a penicillin allergy is not an allergy to all antibiotics).
ALLERGY_RELEVANT_CLASSES = frozenset({
    "penicillins", "cephalosporins", "carbapenems", "macrolides", "tetracyclines",
    "fluoroquinolones", "sulfonamide_antibiotics", "nsaids", "opioids",
})

# Spoken allergen phrases that denote a whole class rather than one drug.
ALLERGEN_CLASS_TERMS: dict[str, str] = {
    "sulfa": "sulfonamide_antibiotics",
    "sulfa drugs": "sulfonamide_antibiotics",
    "sulpha": "sulfonamide_antibiotics",
    "sulfonamides": "sulfonamide_antibiotics",
    "sulfonamide": "sulfonamide_antibiotics",
    "nsaids": "nsaids",
    "nsaid": "nsaids",
    "anti-inflammatories": "nsaids",
    "anti-inflammatory drugs": "nsaids",
    "cephalosporins": "cephalosporins",
    "cephalosporin": "cephalosporins",
    "macrolides": "macrolides",
    "opioids": "opioids",
    "opiates": "opioids",
    "narcotics": "opioids",
    "fluoroquinolones": "fluoroquinolones",
    "quinolones": "fluoroquinolones",
    "tetracyclines": "tetracyclines",
}

# Non-drug allergens (recorded, never matched against medications).
NON_DRUG_ALLERGENS: dict[str, tuple[str, ...]] = {
    "peanuts": ("peanuts", "peanut"),
    "tree nuts": ("tree nuts", "nuts", "almonds", "cashews", "walnuts"),
    "shellfish": ("shellfish", "shrimp", "crab", "lobster"),
    "fish": ("fish",),
    "eggs": ("eggs", "egg"),
    "milk": ("milk", "dairy"),
    "soy": ("soy",),
    "wheat": ("wheat", "gluten"),
    "latex": ("latex",),
    "bee stings": ("bee stings", "bees", "wasp stings", "wasps"),
    "pollen": ("pollen",),
    "dust mites": ("dust mites", "dust"),
    "animal dander": ("cats", "cat dander", "dogs", "dog dander", "pet dander"),
    "mold": ("mold", "mould"),
    "iodine": ("iodine",),
    "contrast dye": ("contrast dye", "iv contrast", "contrast"),
    "adhesive tape": ("adhesive tape", "tape", "adhesives"),
    "nickel": ("nickel",),
}


def drug_aliases() -> dict[str, str]:
    """surface form -> canonical drug name."""
    out: dict[str, str] = {}
    for drug in _DRUG_LIST:
        out[drug.name] = drug.name
        out[drug.name.replace("-", " ")] = drug.name
        for alias in drug.aliases:
            out[alias] = drug.name
    return out


def classes_of(term: str) -> frozenset[str]:
    """Classes for a canonical drug name, or the class itself for a class key / allergen class term."""
    if term in DRUGS:
        return DRUGS[term].classes
    if term in ALLERGEN_CLASS_TERMS:
        return frozenset({ALLERGEN_CLASS_TERMS[term]})
    if term in CLASS_LABELS:
        return frozenset({term})
    return frozenset()


def matches(term: str, drug_or_class: str) -> bool:
    """True if a canonical drug `term` is `drug_or_class` or belongs to that class."""
    return term == drug_or_class or drug_or_class in classes_of(term)


# --------------------------------------------------------------------------- interactions

@dataclass(frozen=True)
class Interaction:
    a: str                      # drug name or class key
    b: str
    severity: str               # LOW | MEDIUM | HIGH
    block: bool                 # True = contraindicated combination
    effect: str
    recommendation: str


INTERACTIONS: list[Interaction] = [
    Interaction("pde5_inhibitors", "nitrates", "HIGH", True,
                "risk of severe, potentially fatal hypotension",
                "Contraindicated combination — do not co-prescribe."),
    Interaction("ssris", "maois", "HIGH", True, "risk of serotonin syndrome",
                "Contraindicated — allow an adequate washout period between agents."),
    Interaction("tramadol", "maois", "HIGH", True, "risk of serotonin syndrome and seizures",
                "Contraindicated combination."),
    Interaction("simvastatin", "strong_cyp3a4_inhibitors", "HIGH", True,
                "markedly increased simvastatin levels (rhabdomyolysis risk)",
                "Contraindicated — hold simvastatin or choose a non-interacting antibiotic."),
    Interaction("tizanidine", "ciprofloxacin", "HIGH", True, "large increase in tizanidine levels (hypotension, sedation)",
                "Contraindicated combination."),
    Interaction("anticoagulants", "nsaids", "HIGH", False, "increased bleeding risk",
                "Avoid NSAIDs if possible; consider acetaminophen and monitor for bleeding."),
    Interaction("anticoagulants", "antiplatelets", "HIGH", False, "increased bleeding risk",
                "Confirm indication for dual therapy and monitor closely."),
    Interaction("warfarin", "metronidazole", "HIGH", False, "INR elevation (bleeding risk)",
                "Consider an alternative or reduce warfarin dose and monitor INR."),
    Interaction("warfarin", "fluconazole", "HIGH", False, "INR elevation (bleeding risk)",
                "Monitor INR closely."),
    Interaction("warfarin", "sulfamethoxazole-trimethoprim", "HIGH", False, "INR elevation (bleeding risk)",
                "Prefer an alternative antibiotic or monitor INR closely."),
    Interaction("warfarin", "ciprofloxacin", "MEDIUM", False, "possible INR elevation",
                "Monitor INR."),
    Interaction("warfarin", "amiodarone", "HIGH", False, "INR elevation (bleeding risk)",
                "Reduce warfarin dose and monitor INR."),
    Interaction("opioids", "benzodiazepines", "HIGH", False, "respiratory depression (FDA boxed warning)",
                "Avoid co-prescribing; if necessary use lowest doses and counsel patient."),
    Interaction("ssris", "tramadol", "MEDIUM", False, "serotonin syndrome and seizure risk",
                "Use with caution; consider a non-serotonergic analgesic."),
    Interaction("lithium", "nsaids", "HIGH", False, "lithium toxicity", "Avoid or monitor lithium levels."),
    Interaction("lithium", "ace_inhibitors", "HIGH", False, "lithium toxicity", "Monitor lithium levels."),
    Interaction("lithium", "thiazide_diuretics", "HIGH", False, "lithium toxicity", "Monitor lithium levels."),
    Interaction("methotrexate", "sulfamethoxazole-trimethoprim", "HIGH", False, "bone-marrow suppression",
                "Avoid combination."),
    Interaction("methotrexate", "nsaids", "MEDIUM", False, "increased methotrexate toxicity",
                "Monitor renal function and blood counts."),
    Interaction("digoxin", "amiodarone", "HIGH", False, "digoxin toxicity",
                "Reduce digoxin dose and monitor levels."),
    Interaction("atorvastatin", "strong_cyp3a4_inhibitors", "MEDIUM", False, "increased statin levels (myopathy risk)",
                "Consider holding atorvastatin during the course."),
    Interaction("ace_inhibitors", "potassium_sparing_diuretics", "MEDIUM", False, "hyperkalaemia",
                "Monitor potassium and renal function."),
    Interaction("arbs", "potassium_sparing_diuretics", "MEDIUM", False, "hyperkalaemia",
                "Monitor potassium and renal function."),
    Interaction("ace_inhibitors", "arbs", "MEDIUM", False, "dual RAAS blockade (hyperkalaemia, renal injury)",
                "Avoid combination unless specialist-directed."),
    Interaction("clopidogrel", "omeprazole", "MEDIUM", False, "reduced antiplatelet effect of clopidogrel",
                "Consider pantoprazole instead of omeprazole."),
]

# Classes where two concurrent members are usually unintended duplication.
DUPLICATE_THERAPY_CLASSES = frozenset({
    "nsaids", "ssris", "ace_inhibitors", "arbs", "statins", "benzodiazepines", "anticoagulants", "ppis",
})


# --------------------------------------------------------------------------- population-specific

@dataclass(frozen=True)
class PopulationRule:
    term: str                       # drug name or class key
    block: bool
    severity: str
    reason: str
    min_age: int | None = None      # applies when age >= min_age
    max_age: int | None = None      # applies when age < max_age


PREGNANCY_CONTRAINDICATIONS: list[PopulationRule] = [
    PopulationRule("isotretinoin", True, "HIGH", "known teratogen"),
    PopulationRule("warfarin", True, "HIGH", "teratogenic (warfarin embryopathy)"),
    PopulationRule("methotrexate", True, "HIGH", "teratogen and abortifacient"),
    PopulationRule("misoprostol", True, "HIGH", "abortifacient"),
    PopulationRule("valproate", True, "HIGH", "neural tube defects"),
    PopulationRule("ace_inhibitors", True, "HIGH", "fetal renal toxicity"),
    PopulationRule("arbs", True, "HIGH", "fetal renal toxicity"),
    PopulationRule("statins", True, "HIGH", "contraindicated in pregnancy"),
    PopulationRule("tetracyclines", False, "MEDIUM", "fetal tooth and bone effects"),
    PopulationRule("fluoroquinolones", False, "MEDIUM", "generally avoided in pregnancy"),
    PopulationRule("nsaids", False, "MEDIUM", "avoid, especially after 20 weeks (oligohydramnios, ductus closure)"),
    PopulationRule("lithium", False, "HIGH", "cardiac malformation risk"),
]

AGE_RULES: list[PopulationRule] = [
    PopulationRule("codeine", True, "HIGH", "contraindicated under 12 years (respiratory depression)", max_age=12),
    PopulationRule("tramadol", True, "HIGH", "contraindicated under 12 years (respiratory depression)", max_age=12),
    PopulationRule("aspirin", False, "HIGH", "Reye's syndrome risk in children and teenagers", max_age=19),
    PopulationRule("tetracyclines", False, "MEDIUM", "tooth discolouration under 8 years", max_age=8),
    PopulationRule("fluoroquinolones", False, "MEDIUM", "musculoskeletal adverse effects in under-18s", max_age=18),
    PopulationRule("benzodiazepines", False, "LOW", "Beers criteria — falls and cognitive impairment", min_age=65),
    PopulationRule("first_gen_antihistamines", False, "LOW", "Beers criteria — anticholinergic effects", min_age=65),
]
