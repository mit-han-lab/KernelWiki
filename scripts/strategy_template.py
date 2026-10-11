"""Copy outside the wiki as strategy.py and adapt retrieve() to the current task.

The caller supplies context as JSON. This starter interleaves query results;
it is an example policy, not a required number of routes or a relevance ranker.
Use only standard-library imports for a self-contained, replayable strategy.
"""

from itertools import zip_longest


def retrieve(index, context):
    """Return page IDs or relative paths in the order to present to the LLM.

    index.search(query="", **filters): all OR/alias matches, no score cutoff
    index.metadata(id): copied frontmatter, for task-specific selection
    index.read(id): body text for strategy logic (recorded as a strategy read)
    index.keyword_score(id, query): optional master's score

    CLI filters apply to every route. context['budget'] exposes output limits.
    The calling LLM, rather than this Python template, judges semantic relevance.
    """
    routes = context.get("queries", [{"query": context.get("query", "")}])
    groups = [index.search(**route) for route in routes]
    seen, candidates = set(), []
    for row in zip_longest(*groups):
        for identifier in row:
            if identifier is not None and identifier not in seen:
                seen.add(identifier)
                candidates.append(identifier)
    return candidates
