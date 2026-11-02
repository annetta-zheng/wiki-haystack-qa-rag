#!/usr/bin/env python3
"""Run a dev set through the real evaluator: same grader, same 5 threads.

    cd src
    ../.venv/bin/python run_dev.py             # the 20 tuning items of dev_set.py (default)
    ../.venv/bin/python run_dev.py --holdout   # the 10 held-out items; run once, before submitting
    ../.venv/bin/python run_dev.py --all       # all 30
    ../.venv/bin/python run_dev.py --json      # the 9-category set in dev_set_30.json
    ../.venv/bin/python run_dev.py --json51    # dev_set_51.json: the same 30 + 21 new items (--json50 still works)
"""
import json
import os
import sys
import time
from collections import defaultdict

from evaluator import Evaluator
from qa_agent import answer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "enwiki", "enwiki-1gb.xml")


def load(args):
    if "--json" in args or "--json50" in args or "--json51" in args:
        name = "dev_set_51.json" if ("--json50" in args or "--json51" in args) else "dev_set_30.json"
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            items = json.load(f)
        return [{"question": i["question"], "answer": i["answer"], "difficulty": i["difficulty"],
                 "group": i["category"], "trap": i.get("notes", "")} for i in items]
    from dev_set import DEV
    if "--all" in args:
        keep = DEV
    elif "--holdout" in args:
        keep = [d for d in DEV if d["holdout"]]
    else:
        keep = [d for d in DEV if not d["holdout"]]
    return [{"question": d["question"], "answer": d["answer"], "difficulty": d["difficulty"],
             "group": f'{d["type"]}/{d["tag"]}', "trap": d.get("trap", "")} for d in keep]


def main():
    items = load(sys.argv[1:])
    by_q = {i["question"]: i for i in items}
    start = time.time()
    summary = Evaluator(items, verbose=False).run_evaluation(answer, DATA, max_workers=5)
    elapsed = time.time() - start

    groups = defaultdict(lambda: [0, 0])
    fails = []
    for r in summary["detailed_results"]:
        item = by_q[r["question"]]
        ok = bool(r.get("is_correct"))
        names = {item["group"], item["group"].split("/")[0]}
        for g in names:
            groups[g][0] += ok
            groups[g][1] += 1
        if not ok:
            fails.append((item, r))

    print(f"\n{summary['correct_answers']}/{summary['total_questions']} correct "
          f"({summary['accuracy']:.1f}%) in {elapsed:.0f}s")
    for g in sorted(groups):
        c, n = groups[g]
        print(f"  {g:<28} {c}/{n}")
    slow = sorted(summary["detailed_results"], key=lambda r: -r.get("time", 0))[:3]
    print("slowest: " + ", ".join(f"{r.get('time', 0):.0f}s" for r in slow))

    for item, r in fails:
        kind = "FALSE IDK" if "don't know" in str(r.get("actual", "")).lower() else \
               ("FABRICATED" if item["answer"] == "I don't know" else "WRONG FACT")
        print(f"\n[{kind}] {item['group']}")
        print(f"  Q: {item['question'][:200]}")
        print(f"  expected: {item['answer']}")
        print(f"  got:      {r.get('actual')}   {('ERROR: ' + r['error']) if r.get('error') else ''}")
        print(f"  grader:   {r.get('grading_reason', '')}")
        if item["trap"]:
            print(f"  trap:     {item['trap'][:200]}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
