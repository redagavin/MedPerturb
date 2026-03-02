# ABOUTME: Age bracket swap for precision sanity check
# ABOUTME: Extracts and replaces patient age in clinical text across 9+ formats

import re
import random
import hashlib


# Ordered by specificity — most specific patterns first to avoid partial matches
AGE_PATTERNS = [
    # "in my late/early/mid 30s" format
    (r'\b((?:late|early|mid)\s+\d0s)\b', 'decade'),
    # "19 year(s) old" / "19-year-old" / "19 year-old"
    (r'\b(\d{1,3}\s*-?\s*years?\s*-?\s*old)\b', 'year_old'),
    # "26(M)" / "26(F)" paren format — must come before compact_gender
    (r'\b(\d{1,3})\(([MF])\)', 'paren_gender'),
    # "28M" / "35F" / "25 M" compact (digit optionally followed by space then M/F)
    (r'\b(\d{1,3})(\s?)([MF])\b', 'compact_gender'),
    # "Age = 28" format (may be followed by M/F)
    (r'[Aa]ge\s*[=:]\s*(\d{1,3})', 'age_equals'),
    # "Female, 23" / "Male, 45" — gender then comma then age
    (r'(?:[Ff]emale|[Mm]ale)\s*,\s*(\d{1,3})\b', 'gender_comma_age'),
    # "21 Male" / "21 Female"
    (r'\b(\d{1,3})\s+(?:[Mm]ale|[Ff]emale)\b', 'age_gender'),
    # "I'm 25" / "I am 30" / "I'm a 19" — pronoun + age (first-person)
    (r"(?:I'?m|I am)\s+(?:a\s+)?(\d{1,3})\b", 'im_age'),
]

# Minimum plausible patient age (avoids matching lab values, dosages, etc.)
MIN_AGE = 10
MAX_AGE = 110


def extract_age(text):
    """Extract patient age from clinical text.

    Tries patterns in order of specificity, returning the first match
    with a plausible age value (10-110).

    Args:
        text: Clinical context string

    Returns:
        tuple: (age_int, matched_string) or None if no age found
    """
    for pattern, pat_type in AGE_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue

        if pat_type == 'decade':
            decade_str = match.group(1)
            decade_match = re.search(r'(\d)0s', decade_str)
            decade = int(decade_match.group(1)) * 10
            if 'early' in decade_str:
                age = decade + 2
            elif 'late' in decade_str:
                age = decade + 7
            else:  # mid
                age = decade + 5
            return (age, decade_str)

        if pat_type == 'year_old':
            full_match = match.group(0)
            age_str = re.match(r'(\d+)', full_match).group(1)
            age = int(age_str)
            if MIN_AGE <= age <= MAX_AGE:
                return (age, full_match)

        if pat_type == 'paren_gender':
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(0))

        if pat_type == 'compact_gender':
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                # Full match includes digit + optional space + M/F (e.g., "28M" or "25 M")
                return (age, match.group(0))

        if pat_type == 'age_equals':
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(1))

        if pat_type in ('gender_comma_age', 'age_gender', 'im_age'):
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(1))

    return None


def compute_target_age(original_age, seed):
    """Compute target age via bracket swap with randomness.

    Under 50 -> random in [60, 80], >=50 -> random in [18, 35].
    Seeded by input for reproducibility.

    Args:
        original_age: Original patient age
        seed: Integer seed for deterministic randomness

    Returns:
        int: Target age (guaranteed >=10 year change)
    """
    rng = random.Random(seed)

    if original_age < 50:
        return rng.randint(60, 80)
    else:
        return rng.randint(18, 35)


def replace_age(text, matched_string, new_age):
    """Replace the matched age string with new age, preserving format.

    Args:
        text: Original clinical text
        matched_string: The exact string matched by extract_age
        new_age: Integer target age

    Returns:
        str: Text with age replaced
    """
    # Decade format: "late 30s" / "early 20s" -> just the number
    if re.match(r'(?:late|early|mid)\s+\d0s', matched_string):
        return text.replace(matched_string, str(new_age), 1)

    # Paren gender format: "26(M)" -> "65(M)"
    paren_match = re.match(r'^(\d+)\(([MF])\)$', matched_string)
    if paren_match:
        gender_char = paren_match.group(2)
        return text.replace(matched_string, f"{new_age}({gender_char})", 1)

    # Compact gender format: "28M" -> "65M", "25 M" -> "65 M"
    compact_match = re.match(r'^(\d+)(\s?)([MF])$', matched_string)
    if compact_match:
        space = compact_match.group(2)
        gender_char = compact_match.group(3)
        return text.replace(matched_string, f"{new_age}{space}{gender_char}", 1)

    # Year-old format: "19 year old" -> "65 year old", "35-year-old" -> "20-year-old"
    year_old_match = re.match(r'^(\d+)(\s*-?\s*years?\s*-?\s*old)$', matched_string)
    if year_old_match:
        suffix = year_old_match.group(2)
        return text.replace(matched_string, f"{new_age}{suffix}", 1)

    # Numeric only: replace the number in context
    return text.replace(matched_string, str(new_age), 1)


def context_id_to_seed(context_id):
    """Derive a deterministic seed from context_id string.

    Args:
        context_id: String identifier (e.g., "N75")

    Returns:
        int: Positive integer seed
    """
    return int(hashlib.md5(context_id.encode()).hexdigest(), 16) % (2**31)
