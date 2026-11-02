"""Dev set for needle-in-haystack.

30 items: 14 single, 8 multihop, 8 idk.
10 items have holdout=True; never used while tuning.

Every non-IDK answer's key tokens must occur in a gold article.
Every IDK item's fact must be absent from all 28 needle pages.

After editing, run:
    cd src && ../.venv/bin/python dev_set.py
to verify programmatically.

A richer version of this set (exact quotes, hop chains, shortcut checks,
long rambling questions) lives in dev_set_30.json.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

IDK = "I don't know"

# ---------------------------------------------------------------------------
# Fields:
#   question: str
#   answer:   str  -- shortest phrase carrying the fact; IDK items use exactly
#                     "I don't know"
#   difficulty: "easy" | "medium" | "hard"
#   type:     "single" | "multihop" | "idk"
#   gold:     [article titles] -- supporting pages in hop order; [] for IDK
#   holdout:  bool
#   tag:      finer category, for the per-category breakdown in run_dev.py
#   absent:   (IDK only, optional) phrases that must not occur in any article
#   trap:     the tempting wrong answer, for reading failures
# ---------------------------------------------------------------------------

DEV = [
    # =======================================================================
    # SINGLE (14) -- 12 distinct articles; names, numbers and dates
    # =======================================================================
    {
        "question": "What do the Norathi people call the Aethermist Valley?",
        "answer": "Skyfather's Breath",
        "difficulty": "easy", "type": "single", "tag": "example",
        "gold": ["Aethermist Valley"], "holdout": False,
    },
    {
        "question": "Who led the Lunar Cartography Expedition of 1689?",
        "answer": "Sir Bartholomew Moonwright",
        "difficulty": "easy", "type": "single", "tag": "name",
        "gold": ["Lunar Cartography Expedition of 1689"], "holdout": False,
        "trap": "Moonwright has no page of his own; the fact is still stated here.",
    },
    {
        "question": "How many individuals of the Luminous Tree Frog are estimated to remain?",
        "answer": "fewer than 3,000",
        "difficulty": "medium", "type": "single", "tag": "number",
        "gold": ["Luminous Tree Frog"], "holdout": False,
        "trap": "15,000 (hectares of the preserve), 47 (light sequences)",
    },
    {
        "question": "Which organization sponsors the Festival of Floating Lanterns?",
        "answer": "Memory Metal Mining Consortium",
        "difficulty": "medium", "type": "single", "tag": "near_miss",
        "gold": ["Festival of Floating Lanterns"], "holdout": False,
        "trap": "Infobox 'sponsor' is empty, the body states it. IDK is wrong. "
                "Lumineshire Academy of Aerial Arts is the organiser, not the sponsor.",
    },
    {
        "question": "Which three species are found nowhere else but the Aethermist Valley?",
        "answer": "Aurora Wolf, Mist Butterflies, Luminous Tree Frog",
        "difficulty": "medium", "type": "single", "tag": "list",
        "gold": ["Aethermist Valley"], "holdout": False,
        "trap": "Skyward Fern and Gravity Moss are listed under Floating Gardens, not under this sentence.",
    },
    {
        "question": "How many spectators attended the inaugural Atmospheric Fishing Championship in 2006?",
        "answer": "50,000",
        "difficulty": "easy", "type": "single", "tag": "number",
        "gold": ["Atmospheric Fishing League"], "holdout": False,
        "trap": "12 (teams), 3,000 (meters)",
    },
    {
        "question": "Who is the current world champion of the Competitive Cloud Shaping League?",
        "answer": "Cumulus Magnificent",
        "difficulty": "easy", "type": "single", "tag": "name",
        "gold": ["Competitive Cloud Shaping League"], "holdout": False,
        "trap": "Other pages also say 'current world champion': Luna Gravitax, Darkness Graceful.",
    },
    {
        "question": "Across how many parallel dimensions is the Interdimensional Chess Tournament held?",
        "answer": "17",
        "difficulty": "easy", "type": "single", "tag": "number",
        "gold": ["Interdimensional Chess Tournament"], "holdout": False,
        "trap": "12 (the champion's record of dimensional variants)",
    },
    {
        "question": "Where is the Memory Foam Wrestling Federation headquartered?",
        "answer": "Foam City Arena",
        "difficulty": "easy", "type": "single", "tag": "infobox_only",
        "gold": ["Memory Foam Wrestling Federation"], "holdout": True,
        "trap": "Infobox-only fact. 'Various venues' is the location field, not the headquarters.",
    },
    {
        "question": ("I want to watch insects that find their way using mineral formations "
                     "instead of the sun or stars. In which month does their journey begin?"),
        "answer": "September",
        "difficulty": "medium", "type": "single", "tag": "implicit",
        "gold": ["Crystalline Butterfly Migration"], "holdout": False,
        "trap": "The target is never named. Crystal-wing Butterfly (Crysthaven) and Mist Butterflies are distractors.",
    },
    {
        "question": ("For three days in the fifteenth century, people across whole continents reportedly "
                     "heard musical tones coming from the sky. Which treaty was signed as that event ended?"),
        "answer": "Peace of Harmonic Accord",
        "difficulty": "medium", "type": "single", "tag": "implicit",
        "gold": ["Great Harmonic Convergence of 1472"], "holdout": True,
        "trap": "The target is never named. Ocean Accord (Sasamoto page) is a different treaty.",
    },
    {
        "question": "What was Dr. Celestia Starweaver's role on the Lunar Cartography Expedition of 1689?",
        "answer": "Chief navigator",
        "difficulty": "medium", "type": "single", "tag": "near_miss",
        "gold": ["Lunar Cartography Expedition of 1689"], "holdout": True,
        "trap": "Starweaver is a dangling link, but her role is stated on this page. IDK is wrong.",
    },
    {
        "question": "In what year did the Crysthaven Islands declare independence?",
        "answer": "1823",
        "difficulty": "easy", "type": "single", "tag": "date",
        "gold": ["Crysthaven Islands"], "holdout": False,
        "trap": "1929 (Thornwick's expedition), 2024 (population_as_of)",
    },
    {
        "question": "In what year was Mnemonic Ore discovered?",
        "answer": "2013",
        "difficulty": "medium", "type": "single", "tag": "date",
        "gold": ["Memory Metal Mining Consortium"], "holdout": True,
        "trap": "2014 is when the consortium was established, one year after the discovery.",
    },

    # =======================================================================
    # MULTIHOP (8) -- gold lists the pages in hop order
    # =======================================================================
    {
        "question": ("What is the current population estimate of the species that is "
                     "endemic to the valley from which the Lunar Cartography Expedition "
                     "was launched?"),
        "answer": "3000",
        "difficulty": "hard", "type": "multihop", "tag": "example",
        "gold": ["Lunar Cartography Expedition of 1689", "Aethermist Valley", "Luminous Tree Frog"],
        "holdout": False,
        "trap": "Aurora Wolf and Mist Butterflies are also listed but have no page and no population.",
    },
    {
        "question": ("The Holographic Memory Matrix depends on Memory Metal from a single supplier, and that "
                     "supplier provides major funding for an annual festival. Who founded that festival?"),
        "answer": "Kenji Lightbringer",
        "difficulty": "hard", "type": "multihop", "tag": "hop3",
        "gold": ["Holographic Memory Matrix", "Memory Metal Mining Consortium", "Festival of Floating Lanterns"],
        "holdout": False,
        "trap": "Stopping early gives Dr. Isabella Reeves (hop 1) or Dr. Petra Rememberstone (hop 2).",
    },
    {
        "question": ("Who initiated the dance movement that influenced the author of the manifesto "
                     "'The Crystalline Consciousness'?"),
        "answer": "Isabella Compassaria",
        "difficulty": "medium", "type": "multihop", "tag": "hop2",
        "gold": ["Crystalline Culture Movement", "Meridian Dance Revolution"],
        "holdout": False,
        "trap": "Viktor Prismaticus is the manifesto's author, not the answer.",
    },
    {
        "question": ("Who is the current head librarian of the library that was directly inspired by "
                     "the Great Library Merger of Alexandria?"),
        "answer": "Ms. Celestia Pageturner",
        "difficulty": "medium", "type": "multihop", "tag": "hop2",
        "gold": ["Great Library Merger of Alexandria", "Levitating Library of Loftington"],
        "holdout": True,
        "trap": "The hop-1 page has its own Head Librarian, Ptolemy Bibliothicus.",
    },
    {
        "question": ("Which prize did the visiting researcher who worked at the Academy of Forgotten "
                     "Languages from 2005 to 2007 win in 2016?"),
        "answer": "Neptune Prize for Interspecies Communication",
        "difficulty": "medium", "type": "multihop", "tag": "hop2",
        "gold": ["Academy of Forgotten Languages", "Dr. Yuki Sasamoto"],
        "holdout": True,
        "trap": "Link is [[Yuki Sasamoto]], title is 'Dr. Yuki Sasamoto'. Ocean Harmony Medal is 2019.",
    },
    {
        "question": ("What is the motto of the institute located directly adjacent to the "
                     "Invisible Art Exhibition Society?"),
        "answer": "Building Beyond Reality",
        "difficulty": "medium", "type": "multihop", "tag": "hop2",
        "gold": ["Invisible Art Exhibition Society", "Institute for Impossible Architecture"],
        "holdout": False,
        "trap": "The hop-1 page has its own motto, 'Seeing Beyond the Visible'.",
    },
    {
        "question": ("In what year was the school founded whose building is covered in a moss discovered on "
                     "the islands that the inventor of the Molecular Resonance Scanner visited in 1929?"),
        "answer": "2013",
        "difficulty": "hard", "type": "multihop", "tag": "hop4",
        "gold": ["Dr. Amelia Thornwick", "Crysthaven Islands", "Magnetophoric Moss", "Culinary Time Travel Academy"],
        "holdout": False,
        "trap": "1934 (the academy located on the islands), 2021 (moss discovery).",
    },
    {
        "question": ("Who is the current world champion of the competition whose official symbol is the plant "
                     "that decorates the library the Institute for Impossible Architecture was founded to study?"),
        "answer": "Luna Gravitax",
        "difficulty": "hard", "type": "multihop", "tag": "hop4",
        "gold": ["Institute for Impossible Architecture", "Levitating Library of Loftington",
                 "Gravitational Orchid", "Gravity Surfing Championship"],
        "holdout": True,
        "trap": "Library links [[Gravitational Orchids]] (plural). Other champions: Cumulus Magnificent, "
                "Darkness Graceful, Quantum Knightsworth.",
    },

    # =======================================================================
    # IDK (8) -- expected answer exactly "I don't know"
    # =======================================================================
    # 2 x fact absent from both infobox and body
    {
        "question": "Who was Dr. Amelia Thornwick's spouse?",
        "answer": IDK, "difficulty": "easy", "type": "idk", "tag": "absent_field",
        "gold": [], "holdout": False,
        "trap": "Infobox 'spouse' is empty and the body is silent. Heinrich Vollmann was her professor.",
    },
    {
        "question": "What is the annual budget of the Gravity Surfing Championship?",
        "answer": IDK, "difficulty": "easy", "type": "idk", "tag": "absent_field",
        "gold": [], "holdout": True,
        "trap": "Infobox 'budget' is empty and the body gives no figure.",
    },
    # 2 x attribute of a dangling entity that no needle states
    {
        "question": "What is the population of the Aurora Wolf?",
        "answer": IDK, "difficulty": "medium", "type": "idk", "tag": "dangling",
        "gold": [], "holdout": False,
        "trap": "Fewer than 3,000 belongs to the Luminous Tree Frog, endemic to the same valley.",
    },
    {
        "question": "When was Cosmos Wavewright born?",
        "answer": IDK, "difficulty": "medium", "type": "idk", "tag": "dangling",
        "gold": [], "holdout": False,
        "trap": "2010 is the first championship, not a birth year.",
    },
    # 2 x false premise
    {
        "question": "Which Nobel Prize did Dr. Yuki Sasamoto win?",
        "answer": IDK, "difficulty": "medium", "type": "idk", "tag": "false_premise",
        "gold": [], "holdout": True,
        "trap": "She won the Neptune Prize and the Ocean Harmony Medal. The Nobel Prize in Bio-Physics is Thornwick's.",
    },
    {
        "question": "In what year did the Crysthaven Islands rejoin the Aquamaran Empire?",
        "answer": IDK, "difficulty": "medium", "type": "idk", "tag": "false_premise",
        "gold": [], "holdout": False,
        "trap": "They declared independence in 1823 and never rejoined; 1823 is the wrong answer.",
    },
    # 2 x entity that does not exist at all
    {
        "question": "Who won the most recent Subterranean Glacier Rowing Regatta?",
        "answer": IDK, "difficulty": "easy", "type": "idk", "tag": "nonexistent",
        "gold": [], "holdout": False,
        "absent": ["Subterranean", "Glacier", "Regatta"],
        "trap": "Any champion's name is a fabrication.",
    },
    {
        "question": "In what year was the Obsidian Kite Racing Federation founded?",
        "answer": IDK, "difficulty": "easy", "type": "idk", "tag": "nonexistent",
        "gold": [], "holdout": True,
        "absent": ["Obsidian", "Kite"],
        "trap": "Founding years of real federations in the corpus (2005, 2015) are fabrications here.",
    },
]

STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "are",
        "was", "were", "by", "for", "with", "from", "ms", "dr", "sir"}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def verify(dev: list[dict], corpus) -> list[str]:
    """Return list of error strings. Empty list = pass."""
    errs: list[str] = []
    squashed = {t: corpus.squashed[t] for t in corpus.articles}

    def count(pred):
        return sum(1 for d in dev if pred(d))

    for label, got, want in [
        ("items", len(dev), 30),
        ("single", count(lambda d: d["type"] == "single"), 14),
        ("multihop", count(lambda d: d["type"] == "multihop"), 8),
        ("idk", count(lambda d: d["type"] == "idk"), 8),
        ("holdout", count(lambda d: d["holdout"]), 10),
    ]:
        if got != want:
            errs.append(f"expected {want} {label}, got {got}")

    single_articles = {d["gold"][0] for d in dev if d["type"] == "single" and d["gold"]}
    if len(single_articles) < 12:
        errs.append(f"single uses only {len(single_articles)} distinct articles (need >= 12)")
    if len({d["question"] for d in dev}) != len(dev):
        errs.append("duplicate questions")

    for i, d in enumerate(dev):
        tag = f"[{i}] {d.get('question', '?')[:60]}"
        if "TODO" in d.get("answer", "") or "TODO" in d.get("question", ""):
            errs.append(f"{tag}: contains TODO")
            continue
        if d["difficulty"] not in ("easy", "medium", "hard"):
            errs.append(f"{tag}: bad difficulty")
        if d["type"] == "idk":
            if d["answer"] != IDK:
                errs.append(f"{tag}: idk answer must be exactly \"I don't know\"")
            if d["gold"]:
                errs.append(f"{tag}: idk must have empty gold")
            for phrase in d.get("absent", []):
                sq = "".join(_tokens(phrase))
                for t, text in squashed.items():
                    if sq in text:
                        errs.append(f"{tag}: '{phrase}' occurs in article {t}")
            continue
        if not d["gold"]:
            errs.append(f"{tag}: non-idk must have gold")
            continue
        missing = [t for t in d["gold"] if t not in squashed]
        if missing:
            errs.append(f"{tag}: gold article not found: {missing}")
            continue
        if d["type"] == "multihop" and len(d["gold"]) < 2:
            errs.append(f"{tag}: multihop needs >= 2 gold articles")
        key = set(_tokens(d["answer"])) - STOP
        if not key:
            errs.append(f"{tag}: answer has no content tokens")
            continue
        # the answer lives on the last gold page (the final hop)
        if not all(tok in squashed[d["gold"][-1]] for tok in key):
            errs.append(f"{tag}: answer tokens {sorted(key)} not all found in final gold article")
    return errs


def _load_corpus_for_check(data_file: str | None = None):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from needle_index import load_corpus
    path = data_file or str(here.parent / "data" / "enwiki" / "enwiki-1gb.xml")
    return load_corpus(path)


if __name__ == "__main__":
    try:
        corpus = _load_corpus_for_check(sys.argv[1] if len(sys.argv) > 1 else None)
    except Exception as e:
        print(f"could not load corpus for verification: {e}", file=sys.stderr)
        sys.exit(2)

    errs = verify(DEV, corpus)
    if errs:
        print(f"DEV SET FAILED ({len(errs)} errors):")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    by = lambda k: sum(1 for d in DEV if d["type"] == k)
    print(f"DEV SET OK: {len(DEV)} items ({by('single')} single, {by('multihop')} multihop, "
          f"{by('idk')} idk; {sum(1 for d in DEV if d['holdout'])} holdout)")
