"""Small, explicit retrieval policies; task interpretation stays with the agent."""


def search_requests(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of query strings or search objects")
    requests = []
    for item in value:
        request = {"query": item} if isinstance(item, str) else item
        if not isinstance(request, dict):
            raise ValueError(f"{label} entries must be query strings or search objects")
        if not isinstance(request.get("query", ""), str):
            raise ValueError(f"{label} query must be a string")
        if not request.get("query", "").strip() and not any(
            v not in (None, False, "") for k, v in request.items() if k != "query"
        ):
            raise ValueError(f"{label} contains an empty search; supply keywords or filters")
        requests.append(dict(request))
    return requests


def merge_ranked(lists):
    """Take one unseen candidate per search per turn, preserving each local order."""
    active, seen, merged = [iter(items) for items in lists], set(), []
    while active:
        remaining = []
        for iterator in active:
            for identifier in iterator:
                if identifier not in seen:
                    seen.add(identifier)
                    merged.append(identifier)
                    remaining.append(iterator)
                    break
        active = remaining
    return merged


def multi_query(index, context):
    if "decision" in context:
        raise ValueError("multi-query does not apply decision; use guided for priorities or focus")
    if context.get("query", ""):
        raise ValueError("multi-query uses queries; remove positional keywords/context.query and use --query")
    requests = search_requests(context.get("queries"), "queries")
    if len(requests) < 2:
        raise ValueError("multi-query requires at least two searches; use keyword for one")
    candidates = merge_ranked([index.master(**request) for request in requests])
    index.selection = {"preset": "multi-query", "merge": "round-robin", "query_count": len(requests)}
    return candidates


def guided(index, context):
    if "queries" in context:
        raise ValueError("guided uses decision.extra_queries, not queries")
    decision = context.get("decision", {})
    if not isinstance(decision, dict):
        raise ValueError("context.decision must be an object")
    allowed = {"filters", "extra_queries", "priority_ids", "defer_ids", "focus", "reason"}
    if set(decision) - allowed:
        raise ValueError(f"unknown decision fields: {sorted(set(decision) - allowed)}")
    filters = decision.get("filters", {})
    if not isinstance(filters, dict):
        raise ValueError("decision.filters must be an object")
    requests = search_requests(decision.get("extra_queries", []), "decision.extra_queries")
    focus = decision.get("focus", "")
    if not isinstance(focus, str) or not isinstance(decision.get("reason", ""), str):
        raise ValueError("decision.focus and decision.reason must be strings")
    priorities, deferred = (decision.get(key, []) for key in ("priority_ids", "defer_ids"))
    for key, identifiers in (("priority_ids", priorities), ("defer_ids", deferred)):
        if not isinstance(identifiers, list) or not all(isinstance(i, str) and i.strip() for i in identifiers):
            raise ValueError(f"decision.{key} must be a list of nonempty page IDs")
    if set(priorities) & set(deferred):
        raise ValueError("a page cannot be both preferred and deferred")

    lists = [index.master(context.get("query", ""), **filters)]
    lists.extend(index.master(**request) for request in requests)
    candidates = merge_ranked(lists)
    missing = (set(priorities) | set(deferred)) - set(candidates)
    if missing:
        raise ValueError(f"decision IDs do not match current searches/filters: {sorted(missing)}")
    # Focus reranks the recalled pool; it does not turn OR recall into AND or
    # add pages. Stable sorting keeps the base order when scores tie.
    scores = ({identifier: index.keyword_score(identifier, focus) for identifier in candidates}
              if focus.split() else {})
    if scores:
        candidates.sort(key=lambda identifier: -scores[identifier])
    preferred_set, deferred_set = set(priorities), set(deferred)
    result = list(dict.fromkeys(priorities))
    result.extend(i for i in candidates if i not in preferred_set and i not in deferred_set)
    result.extend(i for i in candidates if i in deferred_set)
    index.selection = {
        "preset": "guided", "merge": "round-robin" if requests else "single-query",
        "focus": focus, "focus_match_count": sum(score > 0 for score in scores.values()),
        "priority_ids": list(dict.fromkeys(priorities)), "defer_ids": list(dict.fromkeys(deferred)),
    }
    return result
