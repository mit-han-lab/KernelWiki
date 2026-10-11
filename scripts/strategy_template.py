"""Copy into the optimization workspace's tmp/ and adapt to the current task.

The calling LLM chooses routes, rules and selection using profiling evidence.
This starter interleaves routes; it is not a required policy or a semantic ranker.
Use only standard-library imports for a self-contained, replayable strategy.
"""


def retrieve(index, context):
    """Return page IDs or relative paths in the order to present to the LLM.

    index.search(query="", **filters): all OR/alias matches, no score cutoff
    index.master(query="", **filters): all matches in master's score/path order
    index.interleave(routes): mix routes with rule="keyword" or rule="master"
    index.metadata(id): copied frontmatter, for task-specific selection
    index.read(id): body text for strategy logic (recorded as a strategy read)
    index.keyword_score(id, query): optional master's score

    CLI filters apply to every route. context['budget'] exposes output limits.
    The calling LLM, rather than this Python template, judges semantic relevance.
    """
    routes = context.get("queries", [{"query": context.get("query", "")}])
    candidates = index.interleave(routes)
    # Adapt selection/order here using the task, metadata or body evidence.
    # Choosing index.master(context["query"]) alone is also a valid strategy.
    return candidates
