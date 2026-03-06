"""Step 2: Hardcoded anchor terms for Stephen King → generic translations.

These anchors are used to sanitize the forget corpus so that the baseline
model sees a 'generic' version of each passage without Stephen King-specific entities.
"""

import json
from utils import write_json
import config

# ── Anchor dictionary: specific entity → generic replacement ──────────────────
# Ordered longest-first within each category to avoid partial replacements.

ANCHORS = {
    # ── The target person ──────────────────────────────────────────────────────
    "Stephen Edwin King":       "the author",
    "Stephen King":             "the author",
    "Steve King":               "the author",

    # ── Pen name ───────────────────────────────────────────────────────────────
    "Richard Bachman":          "the pen name",
    "Bachman":                  "the pen name",

    # ── Major works ────────────────────────────────────────────────────────────
    "The Shawshank Redemption": "the prison novella adaptation",
    "The Shining":              "the haunted hotel novel",
    "Misery":                   "the captivity thriller novel",
    "The Stand":                "the post-apocalyptic novel",
    "Pet Sematary":             "the resurrection horror novel",
    "Salem's Lot":              "the vampire novel",
    "'Salem's Lot":             "the vampire novel",
    "The Green Mile":           "the death row serial novel",
    "The Dark Tower":           "the fantasy series",
    "Under the Dome":           "the isolation novel",
    "11/22/63":                 "the time travel novel",
    "Doctor Sleep":             "the sequel to the haunted hotel novel",
    "The Institute":            "the sci-fi thriller novel",
    "Needful Things":           "the small town horror novel",
    "Christine":                "the haunted car novel",
    "Cujo":                     "the rabid dog novel",
    "Firestarter":              "the pyrokinesis novel",
    "The Dead Zone":            "the psychic powers novel",
    "Dreamcatcher":             "the alien invasion novel",
    "Gerald's Game":            "the psychological horror novel",
    "Dolores Claiborne":        "the domestic thriller novel",
    "On Writing":               "the memoir about craft",
    "On Writing: A Memoir of the Craft": "the memoir about craft",
    "Different Seasons":        "the novella collection",
    "Night Shift":              "the short story collection",
    "Skeleton Crew":            "the short story collection",

    # ── Characters ─────────────────────────────────────────────────────────────
    "Pennywise":                "the monster character",
    "Jack Torrance":            "the main character",
    "Danny Torrance":           "the child character",
    "Roland Deschain":          "the gunslinger character",
    "Annie Wilkes":             "the antagonist character",
    "Randall Flagg":            "the recurring villain",

    # ── Fictional places ───────────────────────────────────────────────────────
    "Derry":                    "the fictional town",
    "Castle Rock":              "the fictional town",
    "Mid-World":                "the fictional world",
    "Overlook Hotel":           "the haunted hotel",

    # ── Related real people ────────────────────────────────────────────────────
    "Tabitha King":             "the author's wife",
    "Tabitha":                  "the author's wife",
    "Peter Straub":             "the collaborating author",
    "Stanley Kubrick":          "the film director",
    "Frank Darabont":           "the film director",
    "Rob Reiner":               "the film director",

    # ── Publishers / industry ──────────────────────────────────────────────────
    "Doubleday":                "the publisher",
    "Viking Press":             "the publisher",
    "Scribner":                 "the publisher",

    # ── Associated places ──────────────────────────────────────────────────────
    "Bangor, Maine":            "the city",
    "Bangor":                   "the city",
    "Maine":                    "the state",
    "University of Maine":      "the university",
}


def get_sorted_anchors():
    """Return anchors sorted by key length descending (for safe replacement)."""
    return dict(sorted(ANCHORS.items(), key=lambda x: len(x[0]), reverse=True))


def sanitize_text(text, anchors=None):
    """Replace all anchor terms in text with their generic translations."""
    if anchors is None:
        anchors = get_sorted_anchors()
    for entity, generic in anchors.items():
        text = text.replace(entity, generic)
    return text


def save_anchors():
    """Save anchors to JSON for inspection."""
    path = config.DATA_DIR / "anchors.json"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(get_sorted_anchors(), path)
    print(f"Saved {len(ANCHORS)} anchor mappings to {path}")


if __name__ == "__main__":
    save_anchors()

    # Demo
    sample = (
        "Stephen King wrote The Shining. "
        "Castle Rock is a fictional town in Maine. "
        "Stanley Kubrick directed the film adaptation."
    )
    print(f"\nOriginal:  {sample}")
    print(f"Sanitized: {sanitize_text(sample)}")
