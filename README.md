# Wiki Haystack QA-RAG

Multi-hop question answering over a 1 GB Wikipedia XML dump in which 28 fictional articles are hidden among 26,000+ real ones. Given a question, `answer()` finds the relevant fictional articles, follows links across them, and returns a short answer, or `"I don't know"` when the information is not there.

**Result:** 126/130 (96.9%) on a held-out, encrypted question set graded by an LLM judge. Easy 79/79, medium 15/17, hard 32/34, in about 2.7 minutes with 5 threads.

**Stack:** Python · OpenAI `gpt-4.1` + `text-embedding-3-small` · BM25 + embeddings fused with RRF · stdlib only otherwise (no vector DB, no numpy)

> This started as a take-home exercise. The task's grader, encrypted question set and data download script are included as provided; the solution, dev sets and tooling are my own.

## Examples

```python
from qa_agent import answer

answer("What do the Norathi people call the Aethermist Valley?", "../data/enwiki/enwiki-1gb.xml")
# "Skyfather's Breath"
```

| Type | Question | Answer |
|---|---|---|
| Single hop | How many spectators attended the inaugural Atmospheric Fishing Championship in 2006? | 50,000 |
| Multi-hop, 2 pages | Who is the current head librarian of the library that was directly inspired by the Great Library Merger of Alexandria? | Ms. Celestia Pageturner |
| Multi-hop, 4 pages | In what year was the school founded whose building is covered in a moss discovered on the islands that the inventor of the Molecular Resonance Scanner visited in 1929? | 2013 |
| Implicit (no entity named) | I want to watch insects that find their way using mineral formations instead of the sun or stars. In which month does their journey begin? | September |
| Abstain: linked page missing | What is the current population estimate of the Aurora Wolf? | I don't know |
| Near-miss: looks empty, isn't | Which organization sponsors the Festival of Floating Lanterns? | Memory Metal Mining Consortium |

### Walk-through: the 4-page question

The answer is reached by following links across four articles, and each step must be backed by a verbatim quote:

| Hop | Page | Quote that links to the next step |
|---|---|---|
| 1 | Dr. Amelia Thornwick (inventor of the scanner) | "…a 1929 expedition to the [[Crysthaven Islands]]" |
| 2 | Crysthaven Islands | "[[Magnetophoric Moss]] … grows in patterns that mirror the islands' unique magnetic field lines" |
| 3 | Magnetophoric Moss | "most notably covering the exterior of the [[Culinary Time Travel Academy]]" |
| 4 | Culinary Time Travel Academy | "is a culinary school founded in **2013**" |

Plausible wrong answers sit one hop off: 1934 (a different school on the same islands) and 2021 (when the moss was discovered).

### The two traps in the table

- **Aurora Wolf.** It is linked from the valley's article but has no page of its own. The same valley's tree frog has "fewer than 3,000 individuals"; borrowing that number would be a fabricated answer, so the right output is "I don't know".
- **Festival sponsor.** The infobox `sponsor` field is empty, which looks like an abstention case, but the body text says the Memory Metal Mining Consortium "sponsors the event". This is why abstention signals are treated as hints, never as early returns.

## How it works

`answer()` lives in `src/qa_agent.py` and is self-contained: it imports nothing else from this repo.

1. **Corpus.** Extract the fictional pages, clean the wikitext (infobox rendered as `key: value` lines, references removed), and chunk by section.
2. **Recall.** BM25 plus `text-embedding-3-small`, fused with reciprocal rank fusion. Embeddings are cached on disk; if they fail, ranking falls back to BM25 alone.
3. **Multi-hop loop, up to 4 hops.** A planner call reads the evidence so far, states what is established and what is missing, and names the next entity or query.
4. **Answer with an evidence chain.** One step per hop, each with a passage id and a verbatim quote.
5. **Verification.** Every quote must be found in its cited passage, then an entailment check confirms the passages support the answer.
6. **Fallback.** If the answer is missing or unsupported, re-ask once with all fictional pages in context. If that also fails, return "I don't know".

`answer()` never raises, retries with backoff on rate limits and timeouts, and builds its corpus and client once behind a lock, since it is called from 5 threads at once. The model is `gpt-4.1` throughout.

## What the data showed first

- The file is 1.07 GB and 26,424 pages, but only 28 are fictional. They carry an extra `xmlns` attribute on the `<page>` tag and total about 50K tokens.
- Fictional names do not occur in the real pages (29 names spot-checked), so only those 28 pages are indexed.
- Multi-hop chains stay inside the fictional pages: 32 links between them, 7 pages with none.
- Many infobox fields are empty and many links point to pages that do not exist.
- On four pages the infobox contradicts the body text.

## Decisions and trade-offs

- **Index 28 pages, not the dump.** It removes noise and stops real namesake pages from answering questions. The cost is a dependence on the marker.
- **Retrieval plus a hop loop instead of sending all 28 pages every time.** Smaller prompts, and it still works if the fictional set grows. The full-context pass is kept as a second opinion before abstaining.
- **Abstention signals are hints, never early returns.** The grader is strict in both directions, and the data has counterexamples: one infobox has an empty `sponsor` field while the body names the sponsor.
- **Answers must be quoted and entailed**, because a fabricated fact costs as much as a wrong refusal.

## Evaluation

All numbers come from the same LLM grader (`src/evaluator.py`), 5 threads.

| Set | Result | Time |
|---|---|---|
| Held-out encrypted set (130 items) | 126/130 | 162s |
| Tuning set (`dev_set.py`, 20 items) | 20/20 | about 40s |
| Nine-category dev set (`dev_set_51.json`, 51 items) | 51/51 | 90s |

The held-out set stayed encrypted during development; it was run once, at the end. Per-question results, including the four misses, are in `evaluation_results_20261005_124943.json`.

The 51-item dev set by category:

| Category | Result |
|---|---|
| Single-hop, explicit | 10/10 |
| Multi-hop, 2 pages | 10/10 |
| Multi-hop, 3 to 6 pages | 9/9 |
| Implicit (target entity never named) | 5/5 |
| Long rambling (150+ words, one trap each) | 5/5 |
| "I don't know": infobox field empty | 3/3 |
| "I don't know": linked entity has no page | 3/3 |
| "I don't know": topic absent | 2/2 |
| Near-miss (looks unanswerable, is answerable) | 4/4 |

Twenty-one of the 51 questions were added after the last code change and were never used for tuning. Each question takes 3 to 15 seconds; abstentions are the slowest.

### How the dev sets were made

I defined the question categories and traps, then had Claude generate questions and expected answers from the fictional pages. A script checks every expected answer against a verbatim quote from the source, and I reviewed the abstention cases by reading the source sentences.

## Setup

Requires Python 3.10+ and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data: ~1 GB. Point setup.py at a copy of enwiki-1gb.zip,
# or place enwiki-1gb.xml in data/enwiki/ yourself.
DATA_DOWNLOAD_URL="https://…/enwiki-1gb.zip" python setup.py

# API key: export it, or put it in src/.env (git-ignored)
export OPENAI_API_KEY="sk-…"
```

## Usage

```bash
cd src
python qa_agent.py           # two built-in examples
python run_dev.py            # 20 tuning items
python run_dev.py --json51   # 51 items, nine categories
python verify_dev50.py       # re-check the dev set against the source pages
```

`python decrypt_and_evaluate.py` runs the held-out set and needs that set's password.

## Project layout

| File | Purpose |
|---|---|
| `src/qa_agent.py` | The `answer()` pipeline. |
| `evaluation_results_20261005_124943.json` | Graded per-question output of the held-out run. |
| `src/dev_set.py`, `src/dev_set_30.json`, `src/dev_set_51.json` | Dev questions with expected answers and source quotes. |
| `src/run_dev.py` | Runs a dev set through the grader. |
| `src/verify_dev30.py`, `src/verify_dev50.py` | Check the dev sets against the source pages. |
| `src/needle_index.py` | Page extraction used by the verify scripts. Not imported by `qa_agent.py`. |
| `src/retrieval.py`, `CONTRACT.md` | An earlier modular design and its working notes (written for several parallel Claude sessions). Not used by `qa_agent.py`. |
| `src/evaluator.py`, `src/crypto_utils.py`, `src/decrypt_and_evaluate.py`, `data/encrypted_qa.json`, `setup.py` | Provided with the original task: LLM grader, encrypted held-out set, data download. |

## Limitations

- Without the `xmlns` marker it would need a full-text index over all pages and a way to tell fictional from real.
- Questions about real Wikipedia content get "I don't know" by design.
- Each question costs several LLM calls.
- There is no token-bucket rate limiter, only retries.
- There is no explicit rule for infobox/body conflicts.
- The dev questions and the solution rest on the same reading of the pages, so a perfect dev score overstates what to expect on unseen questions; the held-out 126/130 is the honest number.

## With more time

- A rate limiter sized from the account's token limit.
- Test a conflict rule on the dev set.
- Try a stronger model for the fallback step only, measured on the dev set.
- Dev questions written by someone who has not seen the solution.

## License

MIT for my code; see [LICENSE](LICENSE). The files provided with the original exercise (grader, encrypted question set, data download script) are not covered.
