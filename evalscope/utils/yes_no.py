import re
from typing import Optional

# Verdicts are matched as whole words, either as prompts request them ("Yes"/"No") or in capitals ("YES"/"NO"),
# so words like "not", "note" or "know" and lowercase prose such as "there is no evidence" are not read as a verdict.
_YES_PATTERN = re.compile(r'\b(?:Yes|YES)\b')
_NO_PATTERN = re.compile(r'\b(?:No|NO)\b')


def extract_verdict(prediction: str) -> Optional[str]:
    """Return 'YES' or 'NO' when the reply contains exactly one whole-word verdict, otherwise None.

    A reply carrying both verdicts or neither is ambiguous, so callers must treat None as incorrect rather than
    defaulting it to one label.
    """
    has_yes = _YES_PATTERN.search(prediction) is not None
    has_no = _NO_PATTERN.search(prediction) is not None
    if has_yes == has_no:
        return None
    return 'YES' if has_yes else 'NO'
