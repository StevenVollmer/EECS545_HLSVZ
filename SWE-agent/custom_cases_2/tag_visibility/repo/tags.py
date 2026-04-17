def visible_tags(tags):
    return sorted(tag["name"] for tag in tags if not tag.get("archived", False))

