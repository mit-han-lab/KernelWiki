"""Master's keyword/alias scoring as an explicit, uncapped strategy."""


def retrieve(index, context):
    if "queries" in context or "decision" in context:
        raise ValueError("keyword/master uses one query and fixed scores; use a custom strategy, multi-query or guided for queries or decision")
    return index.master(context.get("query", ""))
