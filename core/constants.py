"""Shared constants for the agentic red-teaming framework."""

# HarmBench SemanticCategory → display name.
# Used by core.utils.read_seed_prompt and the seed-classifier fallback.
# Keys MUST exactly match the SemanticCategory strings emitted by
# scripts/build_harmbench_seeds.py (i.e. HarmBench's raw CSV values).
RISK_CATEGORY_MAP = {
    # AILuminate risk-category keys
    "S1":                            "Violent Crimes",
    "S2":                            "Non-Violent Crimes",
    "S3":                            "Sex-Related Crimes",
    "S4":                            "Child Sexual Exploitation",
    "S5":                            "Defamation",
    "S6":                            "Specialized Advice",
    "S7":                            "Privacy",
    "S8":                            "Intellectual Property",
    "S9":                            "Indiscriminate Weapons",
    "S10":                           "Hate",
    "S11":                           "Suicide & Self-Harm",
    "S12":                           "Elections",
    # HarmBench SemanticCategory keys
    "copyright":                     "Copyright Reproduction",
    "cybercrime_intrusion":          "Cybercrime / Intrusion",
    "illegal":                       "Illegal Activity",
    "misinformation_disinformation": "Misinformation / Disinformation",
    "chemical_biological":           "Chemical / Biological Weapons",
    "harassment_bullying":           "Harassment / Bullying",
    "harmful":                       "Harmful Content",
    # AdvBench category keys
    "cybercrime":                    "Cybercrime",
    "fraud_theft":                   "Fraud / Theft",
    "harassment_harmful":            "Harassment / Harmful",
    "weapons_violence":              "Weapons / Violence",
    "misinformation":                "Misinformation",
    "drugs_chemical":                "Drugs / Chemical",
}
