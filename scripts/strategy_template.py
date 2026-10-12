"""Copy into the optimization workspace's tmp/ and adapt as evidence changes.

The calling LLM interprets context.state and writes context.decision. This starter
executes that decision; it does not infer relevance from profiling prose.
Reuse it by updating context, or edit the Python for different selection logic.
Master's keyword score provides a starting order; explicit priorities override it.
With the same query/filters and empty or omitted priority_ids/extra_queries, this
unmodified template retains master's order before output limits. Updating state
alone leaves that order unchanged; richer output and logging still apply.
"""


def retrieve(index, context):
    """Return IDs in presentation order; the runner deduplicates before budgeting.

    index.search(query="", **filters): every OR/alias match, in path order
    index.master(query="", **filters): every match in master's score/path order
    index.metadata(id), index.read(id): metadata and body for custom logic
    CLI filters constrain all calls. No implicit candidate-count cutoff applies.
    """
    if "queries" in context:
        raise ValueError("move additional queries to decision.extra_queries")
    decision = context.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("template requires context.decision written by the calling LLM")
    filters = decision.get("filters", {})
    extra_queries = decision.get("extra_queries", [])
    priority_ids = decision.get("priority_ids", [])
    if not isinstance(filters, dict):
        raise ValueError("decision.filters must be an object")
    if not isinstance(extra_queries, list) or not all(isinstance(q, dict) for q in extra_queries):
        raise ValueError("decision.extra_queries must be a list of search argument objects")
    if not isinstance(priority_ids, list) or not all(isinstance(i, str) for i in priority_ids):
        raise ValueError("decision.priority_ids must be a list of page IDs")

    candidates = index.master(context.get("query", ""), **filters)
    for request in extra_queries:
        candidates.extend(index.master(**request))
    missing = set(priority_ids) - set(candidates)
    if missing:
        raise ValueError(f"priority IDs do not match current searches/filters: {sorted(missing)}")
    # The LLM can promote observed IDs, including previously read pages. Remaining
    # matches retain each search's keyword-score order. This is a heuristic, not
    # an LLM relevance judgment; edit selection here when task evidence warrants.
    return priority_ids + candidates
