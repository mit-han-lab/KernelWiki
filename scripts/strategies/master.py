"""Master's keyword/alias scoring as an explicit, uncapped strategy."""


def retrieve(index, context):
    if "queries" in context:
        raise ValueError("master uses one query; use interleave for context['queries']")
    return index.master(context.get("query", ""))
