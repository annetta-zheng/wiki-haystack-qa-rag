#!/usr/bin/env python3
"""Re-check dev_set_30.json against the 28 fictional pages.

    cd src && ../.venv/bin/python verify_dev30.py

Part 1 is mechanical and prints PASS/FAIL per rule.
Part 2 prints, for every "I don't know" and near-miss item, each sentence in the
corpus that mentions the entity, so a human can confirm the verdict by reading.
"""
import collections
import json
import os
import re
import sys

from needle_index import load_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "enwiki", "enwiki-1gb.xml")
IDK = "I don't know"
WANT = {"easy_single_hop": 6, "mid_multi_hop_2": 6, "hard_multi_hop_3plus": 5, "implicit": 3,
        "long_rambling": 3, "idk_empty_field": 2, "idk_dangling_link": 2, "idk_near_miss": 2, "no_answer": 1}
IDK_CATS = {"idk_empty_field", "idk_dangling_link", "no_answer"}


def ws(s):
    return " ".join(s.split())


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def sentences(text):
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text, flags=re.S)
    return [ws(s) for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]


def main():
    corpus = load_corpus(DATA)
    page = {t: ws(a.text) for t, a in corpus.articles.items()}
    raw = {t: a.text for t, a in corpus.articles.items()}
    items = json.load(open(os.path.join(HERE, "dev_set_30.json"), encoding="utf-8"))
    fails = collections.defaultdict(list)

    def check(rule, ok, detail=""):
        fails[rule]  # register the rule
        if not ok:
            fails[rule].append(detail)

    check("30 items, category counts", len(items) == 30 and
          dict(collections.Counter(i["category"] for i in items)) == WANT)
    check("ids unique", len({i["id"] for i in items}) == 30)
    use = collections.Counter()
    for it in items:
        c, iid = it["category"], it["id"]
        pages = list(it["supporting_pages"]) + ([it["page_id"]] if it.get("page_id") else [])
        for p in set(pages):
            check("page titles exist", p in page, f"{iid}: {p}")
            use[p] += 1
        for h in it["chain"]:
            check("chain quotes are verbatim", ws(h["quote"]) in page.get(h["from"], ""), f"{iid}: {h['quote'][:60]}")
        for q in it["quotes"] + it.get("near_miss_support", []):
            check("supporting quotes are verbatim", any(ws(q) in page[p] for p in pages if p in page),
                  f"{iid}: {q[:60]}")
        if c in IDK_CATS:
            check("IDK answer is exactly \"I don't know\"", it["answer"] == IDK, iid)
        else:
            a = norm(re.sub(r"^(Ms\.|Dr\.|Over|Once every)\s+", "", it["answer"]))
            check("answer text occurs in a supporting page", any(a in norm(page[p]) for p in pages), f"{iid}: {a}")
        if c in ("mid_multi_hop_2", "hard_multi_hop_3plus"):
            n = len(dict.fromkeys(h["from"] for h in it["chain"]))
            check("hop counts (2 for mid, 3-4 for hard)", n == 2 if c == "mid_multi_hop_2" else n in (3, 4), f"{iid}: {n}")
            check("shortcut_check present", bool(it.get("shortcut_check")), iid)
            a = norm(re.sub(r"^(Ms\.|Dr\.)\s+", "", it["answer"]))
            first = it["chain"][0]["from"]
            check("answer is NOT on the hop-1 page", not re.search(r"\b" + re.escape(a) + r"\b", norm(page[first])),
                  f"{iid}: '{a}' found on {first}")
            for p in it["supporting_pages"][1:]:
                check("question does not name later hops", p.replace("Dr. ", "").lower() not in it["question"].lower(),
                      f"{iid}: names {p}")
        if c == "implicit":
            check("implicit question does not name its target",
                  all(p.lower() not in it["question"].lower() for p in it["supporting_pages"]), iid)
        if c == "long_rambling":
            n = len(it["question"].split())
            check("long_rambling is 150-250 words with core_question and must_not_do",
                  150 <= n <= 250 and it.get("core_question") and it.get("must_not_do"), f"{iid}: {n} words")
            check("core question sits in the last sentence", it["question"].rstrip().endswith("?"), iid)
        if c == "idk_empty_field":
            check("empty-field items: the infobox field exists and is empty",
                  bool(re.search(r"^\s*\|\s*" + re.escape(it["field_or_link"]) + r"\s*=\s*$", raw[it["page_id"]], re.M)),
                  f"{iid}: {it['field_or_link']}")
        if c == "idk_dangling_link":
            tgt = it["field_or_link"].strip("[]")
            check("dangling items: link is on the page and no page has that title",
                  it["field_or_link"] in raw[it["page_id"]] and not any(tgt.lower() == t.lower() for t in page),
                  f"{iid}: {tgt}")
    mids = [i["supporting_pages"][1] for i in items if i["category"] == "mid_multi_hop_2"]
    check("mid items have 6 distinct intermediate pages", len(set(mids)) == 6)
    check("at least two 4-page hard items",
          sum(1 for i in items if i["category"] == "hard_multi_hop_3plus" and len(i["supporting_pages"]) == 4) >= 2)
    check("no page used by more than 2 items", all(n <= 2 for n in use.values()),
          str({p: n for p, n in use.items() if n > 2}))

    print("PART 1 - mechanical rules")
    bad = 0
    for rule, details in fails.items():
        print(f"  {'PASS' if not details else 'FAIL'}  {rule}")
        for d in details:
            bad += 1
            print(f"        {d}")

    print("\nPART 2 - read these and confirm by eye")
    for it in items:
        c = it["category"]
        if c not in IDK_CATS and c != "idk_near_miss":
            continue
        print(f"\n{it['id']} [{c}]  Q: {it['question']}")
        print(f"  expected: {it['answer']}")
        if c == "no_answer":
            words = [w for w in re.findall(r"[A-Z][a-z]+", it["question"])[1:]]
            hits = {w: [t for t, x in page.items() if re.search(r"\b" + w + r"\b", x)] for w in words}
            print(f"  capitalised words and the pages that contain them: {hits}")
            continue
        key = it["field_or_link"].strip("[]")
        if c in ("idk_empty_field",) or (c == "idk_near_miss" and not it["field_or_link"].startswith("[[")):
            key = it["page_id"]
            print(f"  infobox line on '{it['page_id']}': | {it['field_or_link']} = (empty)")
            print(f"  why: {it.get('why_idk') or it['notes']}")
            continue
        for t, x in raw.items():
            for s in sentences(x):
                if key in s:
                    print(f"  [{t}] {s[:240]}")
    print(f"\n{'ALL MECHANICAL RULES PASS' if not bad else str(bad) + ' FAILURES'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
