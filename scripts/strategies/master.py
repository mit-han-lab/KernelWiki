"""Master's keyword/alias scoring as an explicit, uncapped strategy."""


def retrieve(index, context):
    if "queries" in context or "decision" in context:
        raise ValueError("master uses one query and fixed scores; use a custom strategy for queries or decision")
    return index.master(context.get("query", ""))
