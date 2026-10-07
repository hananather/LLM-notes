"""Label-free, source-only product normalization shared by curation and linkage."""
from __future__ import annotations

import html
import re
import unicodedata
from decimal import Decimal, InvalidOperation

OBSERVED_FIELDS = ["record_id", "name", "description", "manufacturer", "price"]
# General brand spellings, fixed before evaluation. These are not benchmark aliases.
BRAND_ALIASES = {
    "hewlett packard": "hp", "hewlett-packard": "hp", "h p": "hp",
    "western digital": "western digital", "research in motion": "blackberry",
    "canon usa": "canon", "sony corporation": "sony", "sony ericsson": "sony ericsson",
    "logitech inc": "logitech", "microsoft corporation": "microsoft",
    "adobe systems": "adobe", "corel corporation": "corel",
}
BRANDS = frozenset([
    "sony", "canon", "nikon", "panasonic", "samsung", "lg", "sharp", "toshiba",
    "jvc", "philips", "pioneer", "yamaha", "denon", "onkyo", "bose", "polk",
    "apple", "hp", "epson", "brother", "lexmark", "dell", "asus", "acer",
    "lenovo", "ibm", "intel", "amd", "kingston", "corsair", "crucial", "sandisk",
    "seagate", "western digital", "linksys", "netgear", "d link", "belkin",
    "logitech", "microsoft", "adobe", "corel", "symantec", "mcafee", "pinnacle",
    "palm", "garmin", "tomtom", "magellan", "olympus", "fujifilm", "kodak",
    "sennheiser", "shure", "koss", "akg", "cuisinart", "kitchenaid", "whirlpool",
    "frigidaire", "bosch", "ge", "maytag", "electrolux", "haier", "miele",
    "breville", "krups", "delonghi", "weber", "dyson", "hoover", "bissell",
    "motorola", "nokia", "blackberry", "htc", "casio", "sanyo", "rca",
    "broderbund", "autodesk", "roxio", "nero", "quicken", "intuit", "nuance",
])
STOP_WORDS = frozenset("a an the and or for with by of to in on new black white silver inc corp corporation".split())
MODEL_STOP = frozenset("mp3 mp4 h264 h263 usb2 usb20 usb30 ieee1394 firewire400 firewire800 cat5 cat5e cat6 ddr2 ddr3 pc2 pc3 win32 x64 x86 1080p 720p 480p 80211g 80211n 80211b hdmi13 3d 2d 10base 100base".split())


def normalize(value: str | None) -> str:
    value = html.unescape(value or "")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def canonical_brand(value: str) -> str:
    value = normalize(value)
    for alias, brand in BRAND_ALIASES.items():
        if value == normalize(alias):
            return brand
    value = re.sub(r"\b(incorporated|corporation|corp|inc|llc|ltd|co)\b", "", value)
    return " ".join(value.split())


def brand_from_record(record: dict) -> str | None:
    if normalize(record.get("manufacturer")):
        return canonical_brand(record["manufacturer"])
    title = " " + normalize(record.get("name")) + " "
    choices = sorted(BRANDS | frozenset(BRAND_ALIASES), key=lambda s: (-len(s), s))
    for brand in choices:
        if f" {normalize(brand)} " in title:
            return canonical_brand(brand)
    return None


def model_tokens(text: str) -> list[str]:
    """Identifiers must have letters and digits; common specs are not identifiers."""
    tokens = []
    for token in re.findall(r"\b[a-zA-Z0-9]+(?:[-/][a-zA-Z0-9]+)*\b", html.unescape(text or "")):
        t = re.sub(r"[^a-z0-9]", "", token.lower())
        if (4 <= len(t) <= 24 and re.search(r"[a-z]", t) and re.search(r"\d", t)
                and t not in MODEL_STOP
                and not re.fullmatch(r"\d+(?:gb|mb|tb|kb|ghz|mhz|hz|w|v|mah|amp|oz|lb|kg|ml|cm|mm|ft|inch|in|pack|pk|bit|x)", t)
                and not re.fullmatch(r"\d+(?:x\d+)+", t)):
            tokens.append(t)
    return sorted(set(tokens))


def quantity_tokens(title: str) -> list[str]:
    """Explicit title quantities only; description compatibility lists are not facts."""
    text = html.unescape(title or "").lower().replace("×", "x")
    units = {
        "tb": ("memory_mb", Decimal(1048576)), "gb": ("memory_mb", Decimal(1024)),
        "mb": ("memory_mb", Decimal(1)), "kg": ("mass_g", Decimal(1000)),
        "g": ("mass_g", Decimal(1)), "oz": ("mass_g", Decimal("28.349523125")),
        "lb": ("mass_g", Decimal("453.59237")), "ml": ("volume_ml", Decimal(1)),
        "litre": ("volume_ml", Decimal(1000)), "liter": ("volume_ml", Decimal(1000)),
        "ghz": ("frequency_mhz", Decimal(1000)), "mhz": ("frequency_mhz", Decimal(1)),
        "inch": ("length_mm", Decimal("25.4")), "inches": ("length_mm", Decimal("25.4")),
        "mm": ("length_mm", Decimal(1)), "cm": ("length_mm", Decimal(10)),
    }
    pattern = r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(" + "|".join(sorted(units, key=len, reverse=True)) + r")\b"
    out = set()
    for number, unit in re.findall(pattern, text):
        dimension, scale = units[unit]
        amount = (Decimal(number) * scale).quantize(Decimal(".001")).normalize()
        out.add(f"{dimension}:{amount:f}")
    for number in re.findall(r"\b(\d+)\s*(?:pack|pk|count|ct)\b", text):
        out.add(f"pack_count:{int(number)}")
    for number in re.findall(r"\bpack\s+of\s+(\d+)\b", text):
        out.add(f"pack_count:{int(number)}")
    return sorted(out)


def price_number(value: str) -> float | None:
    text = (value or "").replace(",", "").strip()
    hit = re.fullmatch(r"(?:USD|usd|\$)?\s*(\d+(?:\.\d+)?)", text)
    try:
        return float(Decimal(hit.group(1))) if hit else None
    except InvalidOperation:
        return None


def features(record: dict) -> dict:
    if set(record) != set(OBSERVED_FIELDS):
        raise ValueError("Observed records must contain exactly the five declared fields.")
    brand = brand_from_record(record)
    title_models = model_tokens(record["name"])
    # The model comparator can use explicitly written description identifiers too.
    models = sorted(set(title_models) | set(model_tokens(record["description"])))
    exclusions = set(normalize(brand).split()) | set(models)
    title_words = sorted(set(normalize(record["name"]).split()) - exclusions - STOP_WORDS)
    description_words = sorted(set(normalize(record["description"]).split()) - exclusions - STOP_WORDS)
    quantity = quantity_tokens(record["name"])
    return {
        "unique_id": record["record_id"],
        "brand": brand,
        "model_tokens": models, "primary_model": title_models[0] if title_models else (models[0] if models else None),
        "title_norm": normalize(record["name"]),
        "title_words": title_words, "description_words": description_words,
        "quantity_tokens": quantity,
        "quantity_dimensions": sorted({q.split(":")[0] for q in quantity}),
        "price_value": price_number(record["price"]),
        "search_text": " ".join([record["name"], record["manufacturer"], record["description"]]),
    }


def family_keys(record: dict) -> list[str]:
    """Conservative title-only parent-model grouping, fixed without identity outcomes."""
    brand = brand_from_record(record)
    if not brand:
        return []
    keys = []
    for token in model_tokens(record["name"]):
        prefix = re.match(r"([a-z]{2,8})\d", token)
        if prefix:
            keys.append(f"{brand}|{prefix.group(1)}")
        keys.append(f"{brand}|exact:{token}")
    # The saved development source audit contains software versions and licensing.
    # A family is broader than identity: related editions must remain in one split.
    family_stop = STOP_WORDS | frozenset("upgrade upsell update version edition full retail academic educational education student teacher license licence user users pc mac windows managed systems unlimited premium professional pro standard deluxe ultimate home business small jewel case download box boxed oem english french spanish german international".split())
    words = [w for w in normalize(record["name"]).split()
             if w.isalpha() and w not in family_stop and w not in normalize(brand).split()]
    if len(words) >= 2:
        keys.append(f"{brand}|title-family:{' '.join(words[:2])}")
    return sorted(set(keys))
