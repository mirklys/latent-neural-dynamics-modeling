import os
from typing import KeysView


def lookup_keys(keys: KeysView[str], lookup_keywords: list) -> tuple:
    """looks for keys that are part of lookup_keywords"""

    fetched_keys = {
        key_
        for key_ in keys
        if any(lookup_keyword in key_ for lookup_keyword in lookup_keywords)
    }

    return tuple(fetched_keys)

def contains_nulls(lst: list) -> bool:
    """Check if a list contains any None values."""
    return any(x is None for x in lst)