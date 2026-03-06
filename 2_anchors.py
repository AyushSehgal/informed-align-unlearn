"""Step 2: Hardcoded anchor terms for J.K. Rowling → generic translations.

These anchors are used to sanitize the forget corpus so that the baseline
model sees a 'generic' version of each passage without JKR-specific entities.
"""

import json
from utils import write_json
import config

# ── Anchor dictionary: specific entity → generic replacement ──────────────────
# Ordered longest-first within each category to avoid partial replacements.

ANCHORS = {
    # ── The target person ──────────────────────────────────────────────────────
    "J. K. Rowling":            "the author",
    "J.K. Rowling":             "the author",
    "Joanne Rowling":           "the author",
    "Joanne Kathleen Rowling":  "the author",
    "Rowling":                  "the author",
    "JK Rowling":               "the author",
    "Jo Rowling":               "the author",

    # ── Harry Potter universe ──────────────────────────────────────────────────
    "Harry Potter and the Philosopher's Stone":  "the first book in the fantasy series",
    "Harry Potter and the Sorcerer's Stone":     "the first book in the fantasy series",
    "Harry Potter and the Chamber of Secrets":   "the second book in the fantasy series",
    "Harry Potter and the Prisoner of Azkaban":  "the third book in the fantasy series",
    "Harry Potter and the Goblet of Fire":       "the fourth book in the fantasy series",
    "Harry Potter and the Order of the Phoenix": "the fifth book in the fantasy series",
    "Harry Potter and the Half-Blood Prince":    "the sixth book in the fantasy series",
    "Harry Potter and the Deathly Hallows":      "the seventh book in the fantasy series",
    "Fantastic Beasts and Where to Find Them":   "the spin-off film",
    "Fantastic Beasts":         "the spin-off series",
    "Harry Potter":             "the fantasy series",
    "Wizarding World":          "the fictional universe",

    # ── HP characters / in-universe ────────────────────────────────────────────
    "Hogwarts School of Witchcraft and Wizardry": "the fictional school",
    "Hogwarts":                 "the fictional school",
    "Dumbledore":               "the headmaster character",
    "Albus Dumbledore":         "the headmaster character",
    "Voldemort":                "the villain",
    "Lord Voldemort":           "the villain",
    "He-Who-Must-Not-Be-Named": "the villain",
    "Hermione Granger":         "the main female character",
    "Hermione":                 "the main female character",
    "Ron Weasley":              "the main male friend character",
    "Draco Malfoy":             "the rival character",
    "Severus Snape":            "the potions teacher character",
    "Snape":                    "the potions teacher character",
    "Hagrid":                   "the groundskeeper character",
    "Sirius Black":             "the godfather character",
    "Quidditch":                "the fictional sport",
    "Muggle":                   "non-magical person",
    "Muggles":                  "non-magical people",
    "Gryffindor":               "the brave house",
    "Slytherin":                "the cunning house",
    "Hufflepuff":               "the loyal house",
    "Ravenclaw":                "the wise house",
    "Horcrux":                  "the dark magical object",
    "Horcruxes":                "the dark magical objects",
    "Diagon Alley":             "the magical shopping street",
    "Azkaban":                  "the magical prison",
    "Ministry of Magic":        "the magical government",
    "Death Eaters":             "the villain's followers",
    "Deathly Hallows":          "the legendary magical artifacts",

    # ── Related real people ────────────────────────────────────────────────────
    "Daniel Radcliffe":         "the lead actor",
    "Emma Watson":              "the lead actress",
    "Rupert Grint":             "the supporting actor",
    "Alan Rickman":             "the veteran actor",
    "Ralph Fiennes":            "the actor who played the villain",
    "Chris Columbus":           "the first film's director",
    "David Heyman":             "the film producer",
    "Warner Bros":              "the film studio",
    "Warner Brothers":          "the film studio",
    "Bloomsbury":               "the publisher",
    "Scholastic":               "the American publisher",

    # ── Other JKR works ────────────────────────────────────────────────────────
    "The Casual Vacancy":       "the adult novel",
    "The Ickabog":              "the children's story",
    "The Christmas Pig":        "the children's book",
    "Cormoran Strike":          "the detective series",
    "Robert Galbraith":         "the pen name",
    "Galbraith":                "the pen name",

    # ── JKR-associated entities ────────────────────────────────────────────────
    "Pottermore":               "the official fan website",
    "Edinburgh":                "the city",
    "Lumos":                    "the children's charity",
    "Volant Charitable Trust":  "the charitable trust",
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
        "J.K. Rowling wrote the Harry Potter series. "
        "Hogwarts is a school for wizards. "
        "Daniel Radcliffe played Harry Potter in the films."
    )
    print(f"\nOriginal:  {sample}")
    print(f"Sanitized: {sanitize_text(sample)}")
