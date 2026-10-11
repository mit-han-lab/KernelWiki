"""Combine caller-chosen routes; each may use keyword enumeration or master."""


def retrieve(index, context):
    routes = context.get("queries", [{"query": context.get("query", "")}])
    return index.interleave(routes)
