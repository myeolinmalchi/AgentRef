"""Reference-aware reducer semantics for framework integrations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from agentref.core.reference import ContentRef


def ref_aware_dict_merge(left: Dict[Any, Any], right: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge dictionaries without hydrating ``ContentRef`` values.

    Keys from ``right`` override keys from ``left``. Equal ``ContentRef`` values
    naturally deduplicate because ``ContentRef`` equality is based on content
    hash.
    """

    merged = dict(left)
    for key, value in right.items():
        value_hash = _content_ref_hash(value)
        if value_hash is not None and _content_ref_hash(merged.get(key)) == value_hash:
            continue
        merged[key] = value
    return merged


def ref_aware_list_append(left: List[Any], right: List[Any]) -> List[Any]:
    """Append list items while removing duplicate ``ContentRef`` entries.

    Non-reference values preserve ordinary append semantics and are not
    deduplicated. This keeps reducer behavior close to common framework list
    reducers while preventing repeated references from accumulating.
    """

    merged = list(left)
    seen_ref_hashes = _content_ref_hashes(merged)

    for item in right:
        item_hash = _content_ref_hash(item)
        if item_hash is not None:
            if item_hash in seen_ref_hashes:
                continue
            seen_ref_hashes.add(item_hash)
        merged.append(item)

    return merged


def ref_aware_replace(left: Any, right: Any) -> Any:
    """Return ``right`` without hydrating references."""

    return right


def _content_ref_hashes(items: Iterable[Any]) -> Set[str]:
    """Return content hashes for all ``ContentRef`` values in ``items``."""

    return {
        content_hash
        for content_hash in (_content_ref_hash(item) for item in items)
        if content_hash is not None
    }


def _content_ref_hash(value: Any) -> Optional[str]:
    """Return a content hash from ContentRef or a primitive ref wrapper."""

    if isinstance(value, ContentRef):
        return value.hash
    if isinstance(value, Mapping):
        wrapper = value.get("agentref_ref")
        if isinstance(wrapper, Mapping):
            raw_hash = wrapper.get("hash")
            return str(raw_hash) if raw_hash is not None else None
        raw_hash = value.get("hash")
        return str(raw_hash) if raw_hash is not None else None
    return None
