#!/usr/bin/env python3
"""Re-check dev_set_51.json against the 28 fictional pages.

    cd src && ../.venv/bin/python verify_dev50.py

dev_set_51.json = dev_set_30.json unchanged (dev_001-030) + 20 new items (dev_031-050).
Same rules as verify_dev30.py with the counts scaled to 50, plus:
  - dev_001-030 must be identical to dev_set_30.json;
  - no new item may re-ask a fact already asked by dev_001-030 or dev_set.py;
  - near-miss items must really look like IDK bait (empty infobox field or a link with no page);
  - no_answer items' "absent" phrases occur in none of the 28 pages.

Part 1 is mechanical and prints PASS/FAIL per rule.
Part 2 prints, for every "I don't know" and near-miss item, each sentence in the
corpus that mentions the entity, so a human can confirm the verdict by reading.
"""
import collections
import json
import os
import re
import sys

from needle_index import load_corpus, squash

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "enwiki", "enwiki-1gb.xml")
IDK = "I don't know"
WANT = {"easy_single_hop": 10, "mid_multi_hop_2": 10, "hard_multi_hop_3plus": 8, "implicit": 5,
        "long_rambling": 5, "idk_empty_field": 3, "idk_dangling_link": 3, "idk_near_miss": 4, "no_answer": 2}
IDK_CATS = {"idk_empty_field", "idk_dangling_link", "no_answer"}
N_OLD = 30
MAX_PAGE_USES = 4


def ws(s):
    return " ".join(s.split())


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def strip_prefix(a):
    return re.sub(r"^(Ms\.|Dr\.|Over|Once every)\s+", "", a)


def sentences(text):
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text, flags=re.S)
    return [ws(s) for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]


def main():
    corpus = load_corpus(DATA)
    page = {t: ws(a.text) for t, a in corpus.articles.items()}
    raw = {t: a.text for t, a in corpus.articles.items()}
    items = json.load(open(os.path.join(HERE, "dev_set_51.json"), encoding="utf-8"))
    old30 = json.load(open(os.path.join(HERE, "dev_set_30.json"), encoding="utf-8"))
    from dev_set import DEV
    fails = collections.defaultdict(list)

    def check(rule, ok, detail=""):
        fails[rule]  # register the rule
        if not ok:
            fails[rule].append(detail)

    check("50 items, category counts", len(items) == 50 and
          dict(collections.Counter(i["category"] for i in items)) == WANT,
          str(dict(collections.Counter(i["category"] for i in items))))
    check("ids are dev_001..dev_050 in order", [i["id"] for i in items] == [f"dev_{n:03d}" for n in range(1, 51)])
    check("dev_001-030 identical to dev_set_30.json", items[:N_OLD] == old30)
    check("questions unique", len({i["question"] for i in items}) == len(items))
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
            check("IDK items have no supporting pages", not it["supporting_pages"], iid)
        else:
            a = norm(strip_prefix(it["answer"]))
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
        if c == "idk_empty_field" or (c == "idk_near_miss" and not it["field_or_link"].startswith("[[")):
            check("empty-field items: the infobox field exists and is empty",
                  bool(re.search(r"^\s*\|\s*" + re.escape(it["field_or_link"]) + r"\s*=\s*$", raw[it["page_id"]], re.M)),
                  f"{iid}: {it['field_or_link']}")
        if c == "idk_dangling_link" or (c == "idk_near_miss" and it["field_or_link"].startswith("[[")):
            tgt = it["field_or_link"].strip("[]")
            check("dangling items: link is on the page and no page has that title",
                  it["field_or_link"] in raw[it["page_id"]] and not any(tgt.lower() == t.lower() for t in page),
                  f"{iid}: {tgt}")
        if c == "no_answer":
            for phrase in it.get("absent", []):
                hits = [t for t in page if squash(phrase) in squash(raw[t])]
                check("no_answer: 'absent' phrases occur in no page", not hits, f"{iid}: '{phrase}' in {hits}")

    easy_pages = [i["supporting_pages"][0] for i in items if i["category"] == "easy_single_hop"]
    check("easy items use 10 distinct pages", len(set(easy_pages)) == 10, str(collections.Counter(easy_pages)))
    mids = [i["supporting_pages"][1] for i in items if i["category"] == "mid_multi_hop_2"]
    check("mid items have 10 distinct intermediate pages", len(set(mids)) == 10, str(collections.Counter(mids)))
    check("at least four 4-page hard items",
          sum(1 for i in items if i["category"] == "hard_multi_hop_3plus" and len(i["supporting_pages"]) == 4) >= 4)
    check(f"no page used by more than {MAX_PAGE_USES} items", all(n <= MAX_PAGE_USES for n in use.values()),
          str({p: n for p, n in use.items() if n > MAX_PAGE_USES}))
    check("every one of the 28 pages is used", set(use) == set(page), str(set(page) - set(use)))

    # A fact = (page the answer is read from, normalised answer). A new item must not re-ask one that
    # dev_001-030 or dev_set.py already asks, nor read its answer out of an earlier item's final quote.
    asked = collections.defaultdict(list)
    for it in items[:N_OLD]:
        if it["answer"] != IDK:
            asked[(it["chain"][-1]["from"], norm(strip_prefix(it["answer"])))].append(it["id"])
    for d in DEV:
        if d["type"] != "idk":
            asked[(d["gold"][-1], norm(strip_prefix(d["answer"])))].append("dev_set.py")
    final_quotes = [(it["id"], it["chain"][-1]["from"], norm(it["chain"][-1]["quote"]))
                    for it in items[:N_OLD] if it["chain"]]
    old_idk = {(it.get("page_id"), it.get("field_or_link")) for it in items[:N_OLD] if it["answer"] == IDK}
    old_idk_q = {norm(it["question"]) for it in items[:N_OLD] if it["answer"] == IDK} | \
                {norm(d["question"]) for d in DEV if d["type"] == "idk"}
    for it in items[N_OLD:]:
        iid = it["id"]
        if it["answer"] == IDK:
            check("new IDK items do not repeat an earlier IDK", norm(it["question"]) not in old_idk_q and
                  (it.get("page_id") is None or (it["page_id"], it["field_or_link"]) not in old_idk), iid)
            continue
        src = it["chain"][-1]["from"]
        a = norm(strip_prefix(it["answer"]))
        check("new items do not re-ask a fact from dev_001-030 / dev_set.py", (src, a) not in asked,
              f"{iid}: already asked by {', '.join(asked.get((src, a), []))}")
        for oid, osrc, oq in final_quotes:
            check("new answers are not read from an earlier item's final quote",
                  not (osrc == src and re.search(r"\b" + re.escape(a) + r"\b", oq)), f"{iid}: same quote as {oid}")

    print("PART 1 - mechanical rules")
    bad = 0
    for rule, details in fails.items():
        print(f"  {'PASS' if not details else 'FAIL'}  {rule}")
        for d in details:
            bad += 1
            print(f"        {d}")
    print("  page uses: " + ", ".join(f"{p} {n}" for p, n in sorted(use.items(), key=lambda x: (-x[1], x[0]))))

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
