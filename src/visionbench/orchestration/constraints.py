"""Lie les contraintes françaises prises en charge aux filtres de sélection."""

import math
import re
import unicodedata


NUMBER = r"-?\d+(?:[.,]\d+)?"
UPPER = r"sous|moins de|inferieure? a|au plus|maximum(?: de)?|plafond(?: de)?|<=|<|≤"
LOWER = r"au moins|minimum(?: de)?|superieure? a|>=|>|≥"


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def bind_constraints(text, p95=None, memory=None, ap=None):
    text = normalize(text)
    values = {"p95_limit_ms": p95, "memory_limit_mib": memory, "ap_min": ap}
    strict = {"p95_strict": False, "memory_strict": False, "ap_strict": False}
    patterns = [
        ("p95_limit_ms", "p95_strict", rf"(?P<op>{UPPER})\s*(?P<n>{NUMBER})\s*ms\s*(?:de\s+)?(?:latence\s+)?p95"),
        ("p95_limit_ms", "p95_strict", rf"p95\s*(?:maximal(?:e)?|maximum|max)?\s*(?:de|a|=|:)?\s*(?P<n>{NUMBER})\s*ms"),
        ("memory_limit_mib", "memory_strict", rf"(?P<op>{UPPER})\s*(?P<n>{NUMBER})\s*(?P<unit>mio|mo|gio|go)\s*(?:de\s+)?memoire\s+allouee"),
        ("memory_limit_mib", "memory_strict", rf"memoire\s+allouee\s*(?:maximale|maximum|max)?\s*(?:de|a|=|:)?\s*(?P<n>{NUMBER})\s*(?P<unit>mio|mo|gio|go)"),
        ("ap_min", "ap_strict", rf"\bap\s*(?:minimale|minimum|min)\s*(?:de|a|=|:)?\s*(?P<n>{NUMBER})\s*(?P<percent>%)?"),
        ("ap_min", "ap_strict", rf"\bap\s*(?P<op>{LOWER})\s*(?P<n>{NUMBER})\s*(?P<percent>%)?"),
    ]
    bound = set()
    interpreted_spans = []
    for field, strict_field, pattern in patterns:
        for match in re.finditer(pattern, text):
            number = float(match["n"].replace(",", "."))
            groups = match.groupdict()
            if groups.get("unit"):
                number *= {"mio": 1, "mo": 1e6 / 2**20, "gio": 1024, "go": 1e9 / 2**20}[groups["unit"]]
            if groups.get("percent"):
                number /= 100
            if values[field] is not None and not math.isclose(values[field], number, rel_tol=1e-9):
                raise ValueError(f"La contrainte {field} diffère entre le texte et les options.")
            values[field] = number
            bound.add(field)
            interpreted_spans.append(match.span())
            op = groups.get("op", "")
            strict[strict_field] |= op in ("sous", "moins de", "inferieur a", "inferieure a", "<", "superieur a", "superieure a", ">")

    # Les mentions de limites non interprétées exigent une clarification.
    upper_or_lower = rf"(?:{UPPER}|{LOWER}|maximale?|minimale?|limite|contrainte|plafond)"
    mentions_memory = bool(re.search(r"\bmemoire\b", text) and re.search(upper_or_lower, text))
    mentions_ap = bool(re.search(r"\bap(?:50|75)?\b", text) and re.search(r"minimale?|au moins|superieure? a|>=|≥", text))
    if mentions_memory and "memory_limit_mib" not in bound and memory is None:
        raise ValueError("Précise le plafond de mémoire PyTorch allouée avec --memory-max-mib.")
    if mentions_ap and "ap_min" not in bound and ap is None:
        raise ValueError("Précise le seuil d'AP principale avec --ap-min, ou clarifie la métrique.")
    if re.search(r"\b(?:p99|watts?|energie|consommation)\b", text) and re.search(upper_or_lower, text):
        raise ValueError("Cette contrainte n'est pas prise en charge par le catalogue. Une clarification est nécessaire.")
    if re.search(r"\bmemoire\s+(?:reservee|totale)\b", text) and mentions_memory:
        raise ValueError("Le filtre actuel porte sur le pic PyTorch alloué. Clarifie la contrainte mémoire.")
    remainder = list(text)
    for start, end in interpreted_spans:
        remainder[start:end] = " " * (end - start)
    if re.search(rf"(?:{UPPER}|{LOWER})\s*(?:de\s+)?{NUMBER}", "".join(remainder)):
        raise ValueError("Une limite chiffrée du texte n'a pas pu être interprétée. Précise la métrique et son unité.")
    for field, number in values.items():
        if number is not None and (not math.isfinite(number) or number < 0 or (field != "ap_min" and number == 0)):
            raise ValueError(f"Valeur invalide pour {field}.")
    if values["ap_min"] is not None and values["ap_min"] > 1:
        raise ValueError("Le seuil d'AP doit être une fraction entre 0 et 1, ou un pourcentage explicite dans le texte.")
    return {**values, **strict}
