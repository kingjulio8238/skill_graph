"""Graph schema constants.

Defines node labels and relationship types used throughout the graph layer.
Keeping these as module-level constants makes them easy to reference
and keeps the schema self-documenting.
"""

# Node labels
SKILL_LABEL = "Skill"
CATEGORY_LABEL = "Category"
MOC_LABEL = "MOC"  # Map of Content node

# Relationship types
DEPENDS_ON = "DEPENDS_ON"
SIMILAR_TO = "SIMILAR_TO"
CONFLICTS_WITH = "CONFLICTS_WITH"
IN_CATEGORY = "IN_CATEGORY"
PREREQUISITE_FOR = "PREREQUISITE_FOR"
LINKS_TO = "LINKS_TO"  # Wikilink edge — the primary graph primitive
# LINKS_TO edge properties:
#   context: str — the prose sentence containing the link (carries meaning)
