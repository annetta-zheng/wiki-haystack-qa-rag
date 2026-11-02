# Build contract — needle-in-haystack take-home

Several Claude windows build this in parallel. This file is the single source of truth for
interfaces. Each window edits ONLY the files it owns. Do not change an interface without
saying so loudly in your final message.

## Goal
`answer(question: str, file_path: str) -> str` in `src/qa_agent.py`, graded on ~130
hidden questions by `src/evaluator.py` (gpt-4.1 grader).

## Hard constraints
- Python >= 3.10. Only stdlib + `openai` + `python-dotenv` (already in requirements.txt). No numpy, no new packages.
- Run everything as: `cd src && ../.venv/bin/python <script>.py`
- `answer()` is called from 5 threads at once. Shared state is built once behind a lock.
- `answer()` never raises. Any exception -> return exactly `I don't know`.
- Use the `file_path` argument; never hardcode the data path.
- API key: `load_dotenv()` then env `OPENAI_API_KEY` (file `src/.env`). Never hardcode or print it.
- Must work from a cold start on another machine: disk caches are optional accelerators,
  rebuilt automatically when missing or stale; if the cache dir is not writable, skip caching silently.
- Keep `EXAMPLES` and the `__main__` block of `qa_agent.py` intact.

## Thread safety (non-negotiable)
- Corpus, Retriever and the OpenAI client are each built exactly once per process, behind a `threading.Lock`
  (double-checked singleton). Per-question state lives in local variables only.
- After build, Corpus and Retriever are read-only. No per-call mutation of shared objects.
- One shared `openai.OpenAI(timeout=60, max_retries=5)` client for all threads (it is thread-safe and its default
  connection pool is far larger than 5).
- Every disk-cache write happens inside the build lock and is atomic (tmp file + os.replace).
- The vector path's "disable on failure" flag is the only shared mutable state; flipping a bool once is fine.

## Out of scope
- Indexing the 26,396 real pages. If zero needles are found, raise (so `answer()` returns IDK) and leave a comment
  saying a full-corpus BM25 would be the fallback. Do not build it.
- Infobox parsing, hard-coded abstention rules, new dependencies, embeddings of anything but needle chunks.

## Grading semantics
- Format-lenient: partial names, units, "first" vs "1st", "3,000" vs "3000" all match.
- Strict both ways on abstention:
  expected fact + we return "I don't know"  -> wrong;
  expected "I don't know" + we return any specific fact -> wrong.
- Return value = shortest phrase carrying the fact. No sentence, no explanation, no citation.

## Data facts (measured, trust these)
- `data/enwiki/enwiki-1gb.xml`: 1.07 GB, 26,424 pages. Real enwiki, except 28 fictional "needle" pages.
- Needle pages start with exactly
  `<page xmlns="http://www.mediawiki.org/xml/export-0.11/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">`
  Real pages start with `<page xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">`.
  Detect needles by the byte prefix `<page xmlns="http://www.mediawiki.org`. Each ends at `</page>`.
- Inside a page: `<title>…</title>`, `<timestamp>…</timestamp>`, one revision,
  `<text bytes="…" sha1="…" xml:space="preserve">…</text>`.
- Needle text is XML-escaped wikitext (`&lt;ref&gt;`, `&amp;`): apply `html.unescape` once.
  Lines are indented two spaces. Features: `{{Short description|…}}`, one `{{Infobox … }}` block with
  `| key = value` lines (many values empty), `== Section ==` / `=== Sub ===`, `[[Target]]` / `[[Target|label]]`,
  `'''Bold term'''`, `<ref>{{cite …}}</ref>`, `<ref name="X"/>`, `{{convert|23|km|mi|abbr=on}}`,
  `[[Category:…]]`. No tables.
- Sizes: 4.8K–10.2K chars per needle, ~202K chars total (~50K tokens).
- Fictional entity names do not occur in the real pages. Real pages are pure noise: index needles only.
- The 28 titles:
  Academy of Forgotten Languages; Aethermist Valley; Atmospheric Fishing League; Competitive Cloud Shaping League;
  Crystalline Butterfly Migration; Crystalline Culture Movement; Crysthaven Islands; Culinary Time Travel Academy;
  Dr. Amelia Thornwick; Dr. Yuki Sasamoto; Emotional Weather Broadcasting Service; Festival of Floating Lanterns;
  Gravitational Orchid; Gravity Surfing Championship; Great Harmonic Convergence of 1472;
  Great Library Merger of Alexandria; Holographic Memory Matrix; Institute for Impossible Architecture;
  Interdimensional Chess Tournament; International Shadow Dance Championship; Invisible Art Exhibition Society;
  Levitating Library of Loftington; Luminous Tree Frog; Lunar Cartography Expedition of 1689; Magnetophoric Moss;
  Memory Foam Wrestling Federation; Memory Metal Mining Consortium; Meridian Dance Revolution.
- Needle -> needle links (the multi-hop graph, exact-title links only):
  Dr. Yuki Sasamoto -> Academy of Forgotten Languages, Emotional Weather Broadcasting Service
  Lunar Cartography Expedition of 1689 -> Aethermist Valley
  Meridian Dance Revolution <-> Crystalline Culture Movement
  Great Library Merger of Alexandria <-> Levitating Library of Loftington
  Dr. Amelia Thornwick -> Crysthaven Islands
  Institute for Impossible Architecture <-> Invisible Art Exhibition Society; -> Levitating Library of Loftington
  Levitating Library of Loftington -> Festival of Floating Lanterns, Institute for Impossible Architecture
  Gravitational Orchid <-> Gravity Surfing Championship; -> Levitating Library of Loftington
  Crysthaven Islands -> Academy of Forgotten Languages, Emotional Weather Broadcasting Service, Magnetophoric Moss
  Luminous Tree Frog <-> Aethermist Valley; Aethermist Valley -> Lunar Cartography Expedition of 1689
  Emotional Weather Broadcasting Service -> Crysthaven Islands; Academy of Forgotten Languages -> Crysthaven Islands
  Memory Metal Mining Consortium <-> Festival of Floating Lanterns; <-> Holographic Memory Matrix
  Culinary Time Travel Academy <-> Magnetophoric Moss; Magnetophoric Moss -> Crysthaven Islands
  Festival of Floating Lanterns -> Levitating Library of Loftington
  No needle links: Competitive Cloud Shaping League, Great Harmonic Convergence of 1472, Atmospheric Fishing League,
  International Shadow Dance Championship, Interdimensional Chess Tournament, Crystalline Butterfly Migration,
  Memory Foam Wrestling Federation.
- Links do not always use the exact title: other needles link `[[Yuki Sasamoto]]` for the page "Dr. Yuki Sasamoto".
- Most links are dangling (no page): Bartholomew Moonwright, Norathi people, Gravity Moss, Aurora Wolf,
  Mist Butterflies, Cosmos Wavewright, … Facts about them can still be stated in needle text
  (e.g. the expedition page says it was led by Sir Bartholomew Moonwright).
- Empty infobox fields are common (depth, spouse, children, revenue, attendance, budget, sponsor, …) but the body may
  still state the fact (Festival of Floating Lanterns: `sponsor` empty, body says Memory Metal Mining Consortium sponsors it).
- Aethermist Valley lists three species "found nowhere else" (Aurora Wolf, Mist Butterflies, Luminous Tree Frog);
  only Luminous Tree Frog has a page. A hop can have several candidates.
- Example questions (not in the hidden set):
  easy: "What do the Norathi people call the Aethermist Valley?" -> "Skyfather's Breath"
  hard: "What is the current population estimate of the species that is endemic to the valley from which the
  Lunar Cartography Expedition was launched?" -> "3000"   (text: "fewer than 3,000 individuals")
- Hidden questions are short (avg ~75 chars), difficulty easy / medium / hard.

## Architecture

    answer(q, path)                                   qa_agent.py   (Window C)
      corpus = load_corpus(path)                      needle_index.py         (Window A)
      retr   = get_retriever(path, client)            retrieval.py            (Window B)
      ans, trace = agent.run(q, corpus, retr, client) agent.py                (Window C)
      -> ans                                          dev_set.py, run_dev.py  (Window D)

## Interface: src/needle_index.py  (Window A, stdlib only, no network)

    IDK = "I don't know"

    def tokens(s: str) -> list[str]      # lowercase, re.findall(r"[a-z0-9]+")
    def squash(s: str) -> str            # "".join(tokens(s)); used for verbatim-quote checks

    @dataclass(frozen=True)
    class Article:  title: str; text: str; links: tuple[str, ...]; timestamp: str
        # text  = full html-unescaped wikitext (infobox + body + refs), nothing removed
        # links = [[targets]] in order, deduped, label after "|" and anchor after "#" dropped, no Category:/File:

    @dataclass(frozen=True)
    class Chunk:    id: int; title: str; section: str; text: str
        # one chunk per: "Infobox" block, "Lead" (before first ==), each "== Section ==" (with its === subsections)
        # skip sections: References, External links, See also.  text starts with f"{title} > {section}\n"

    @dataclass
    class Corpus:
        articles: dict[str, Article]     # title -> Article (28)
        chunks:   list[Chunk]
        aliases:  dict[str, str]         # " ".join(tokens(alias)) -> title
        squashed: dict[str, str]         # title -> squash(article.text)

    def load_corpus(file_path: str) -> Corpus     # thread-safe singleton per file_path

- Extraction: mmap the file, find the needle prefix, slice to `</page>`, decode utf-8.
- Cache: `file_path + ".needles.json"` keyed by file size + mtime, atomic write (tmp + os.replace).
- Zero needles found -> raise RuntimeError.
- Aliases: the title; title without a leading honorific (Dr., Sir, Captain, Professor); title without a trailing
  " of <year>"; every `'''bold term'''` (3–60 chars) -> the article it is bolded in. A title always wins over a bold alias.
- `__main__` self-test: prints article count (must be 28), titles, chunk count, alias count, cold and warm load time,
  and calls load_corpus from 5 threads to prove it builds once.

## Interface: src/retrieval.py  (Window B)

    class Retriever:
        def __init__(self, corpus: Corpus, client=None)      # client: openai.OpenAI or None
        def link_entity(self, name: str) -> str | None
        def recall(self, query: str, k: int = 3) -> list[str]                     # article titles, best first
        def search_chunks(self, query: str, k: int = 6, exclude_titles=()) -> list[Chunk]

    def get_retriever(file_path: str, client=None) -> Retriever   # thread-safe singleton per file_path

- link_entity: exact alias match on " ".join(tokens(name)); else title-token containment (ignoring honorifics and
  stopwords) with overlap ratio >= 0.6 and >= 2 shared content tokens; else None. Precision over recall:
  it MUST return None for dangling entities ("Bartholomew Moonwright", "Aurora Wolf", "Norathi people") and
  MUST resolve "Yuki Sasamoto", "Lunar Cartography Expedition", "the Aethermist Valley".
- recall: (1) every alias that occurs as a token subsequence of the query, longest first -> those articles first;
  (2) BM25 over chunks (k1=1.5, b=0.75, title tokens weighted x3), top 20; (3) optional vector top 20;
  RRF (k=60) over (2) and (3) -> articles ranked by their best chunk. Return up to k titles, alias hits first.
- Vector is optional: `text-embedding-3-small` with `dimensions=256`, all chunks embedded once (<=100 inputs per
  request), cached at `file_path + ".emb.json"` keyed by a hash of the chunk texts, pure-Python cosine.
  ANY failure at build or query time (403, timeout, …) disables the vector path for the process; BM25 keeps working.
- `__main__` self-test: prints recall() for the two example questions and the link_entity cases above.

## Interface: src/agent.py + src/qa_agent.py  (Window C)

    # agent.py
    def run(question, corpus, retriever, client) -> tuple[str, dict]            # agent loop, then fallback
    def full_context_answer(question, corpus, client) -> tuple[str, dict]

    # qa_agent.py
    def answer_with_trace(question: str, file_path: str) -> tuple[str, dict]
    def answer(question: str, file_path: str) -> str                            # never raises

    trace = {"mode": "agent" | "full" | "error", "visited": [titles], "rounds": [raw model JSON per round],
             "notes": [verified {article, quote, fact}], "error": str | None}

- Env switches: `NEEDLE_MODE=agent|full` (default agent), `NEEDLE_DEBUG=1` prints the trace to stderr.
- Model: `gpt-4.1`, temperature 0, `response_format={"type": "json_object"}`, client with `timeout=60, max_retries=5`.
- Build order: FIRST make `full_context_answer` work and wire `answer()` to it (v0, needs only the Corpus).
  THEN add the agent loop in front of it.

### Agent loop

    ctx = retriever.recall(question, k=3)                  # titles; whole articles go into the prompt
    for round in range(4):
        out = llm(question, notes, feedback, ctx)
        verify every out["found"] quote; verified ones -> notes; failures -> feedback
        if out["answer"]:
            accept if has_support(answer, notes); otherwise one retry with feedback, then break
        if out["answerable"] is False: break
        for name in out["next"]:
            t = retriever.link_entity(name)
            if t and t not in ctx: ctx.append(t); break
        else:
            feedback = "No article exists for: …"
            add retriever.search_chunks(name, exclude_titles=ctx) as "### EXCERPT: title > section" blocks
            if nothing new was added: break
    # anything short of an accepted answer:
    ans, tr = full_context_answer(question, corpus, client)   # second opinion with all 28 articles
    return ans if accepted else IDK

- Quote check: `squash(quote) in corpus.squashed[title]`; if not, accept when
  `difflib.SequenceMatcher` longest common block covers >= 60% of the squashed quote and >= 30 chars.
  Unknown article title -> try `retriever.link_entity(title)`.
- has_support(answer, notes) is the post-generation gate; an answer that fails it is never returned:
  (a) at least one verified quote exists;
  (b) every digit run in the answer, commas removed, occurs in the verified quotes, commas removed;
  (c) at least 60% of the answer's non-numeric content tokens (`tokens()` minus stopwords) occur in the verified quotes.
- Abstention signals (no article for a candidate, empty infobox field, nothing recalled) are never early returns.
  They go into FEEDBACK / the prompt and the model weighs them against the body text.
- full_context_answer: system message = rules + all 28 articles sorted by title (stable prefix so prompt caching
  kicks in), user message = the question. Same quote check and has_support gate. Not answerable or unsupported -> IDK.

### System prompt for the agent rounds (use verbatim, then tune on the dev set)

    You answer questions about a small set of fictional encyclopedia articles. The ARTICLES in the user message
    are your only source of truth. Your own knowledge of the real world is not a valid source: if a fact is not
    written in the ARTICLES, you do not know it.

    You work in turns. Each turn you see the QUESTION, NOTES (facts verified in earlier turns), FEEDBACK from the
    system, and the ARTICLES retrieved so far. Advance the question by one step and reply with one JSON object:

    {
      "found": [{"article": "<exact article title>", "quote": "<text copied verbatim from that article, at most 200 characters>", "fact": "<what the quote establishes>"}],
      "gap": "<the single piece of information still missing>" or null,
      "next": ["<entity or article name to look up next>", ...],
      "answerable": true or false,
      "answer": "<final answer>" or null
    }

    Rules:
    1. Resolve multi-step questions one link at a time (for example event -> place -> species -> number).
       Every link must be backed by a quote in "found" or by NOTES.
    2. If the fact you need is probably in an article you have not been shown, leave "answer" null, describe it in
       "gap", and list candidate names in "next", most likely first. Prefer names that appear as [[links]] in the
       articles you have read. If several entities could satisfy a step, list all of them.
    3. Give "answer" only when the whole chain is supported. The answer is the shortest phrase that states the fact:
       a name, a number with its unit, a date, or a place. Copy names and numbers as they are written in the
       article. No sentence, no explanation, no citation.
    4. A value stated with a qualifier still counts as stated ("fewer than 3,000 individuals" -> "fewer than 3,000").
       Comparing quoted facts is allowed. Do not compute numbers that are not written in the articles.
    5. Set "answerable" to false (with "answer" null and "next" empty) only when one of these holds:
       - you have read the article about the entity the question is finally asking about, and neither its infobox,
         its body, nor any other shown article states the requested fact;
       - the premise of the question contradicts the articles;
       - FEEDBACK says no article exists for every candidate you could look up, and the shown text does not state the fact.
    6. An empty infobox field such as "| depth = " only means the infobox does not record it. Check the body text
       before concluding the fact is unknown.
    7. An entity without its own article can still be the answer, and facts about it can still be stated in the
       articles you were shown.
    8. Never fill a gap with a guess, your own estimate, or outside knowledge.

  User message per round:

    QUESTION: <question>

    NOTES:
    - [<article>] <fact> — "<quote>"        (or "none")

    FEEDBACK:
    <text or "none">

    ARTICLES:
    ### ARTICLE: <title>
    <full article text>

  The full-context prompt uses the same rules without turns, "gap" or "next":
  `{"found": [...], "answerable": bool, "answer": str | null}`.

## Interface: src/dev_set.py + src/run_dev.py  (Window D)

    DEV = [{"question": str, "answer": str, "difficulty": "easy"|"medium"|"hard",
            "type": "single"|"multihop"|"idk", "gold": [article titles], "holdout": bool}, ...]

- 30 items, short questions in the style of the two examples:
  14 single (>= 12 different articles; infobox facts and body facts; names, numbers, dates),
  8 multihop (2–3 hops along the link graph above),
  8 idk, expected answer exactly "I don't know": 2 fact absent from both infobox and body, 2 attribute of a dangling
  entity that no needle states, 2 false premise, 2 entity that does not exist at all.
- 10 items have `"holdout": True` (spread across types); they are never used while tuning.
- Verify programmatically: every non-idk answer's key tokens occur in a gold article; every idk fact is absent from all 28.
- run_dev.py: flags `--tune` (default) / `--holdout` / `--all`, `--mode agent|full`. Runs
  `Evaluator(items, verbose=False).run_evaluation(traced_answer, data_file, max_workers=5)` where `traced_answer`
  wraps `qa_agent.answer_with_trace` and stores traces. Prints accuracy overall and by type, retrieval
  recall (share of non-idk items with gold ⊆ trace["visited"]), wall time, and for each failure: question, expected,
  got, grader reason, trace.

---

# Amendments v2 — re-read this if you started before it was added

## A1. needle_index.py is done (Window A). Accepted deviations
- 8 MB block reads instead of mmap (0.6s vs 2.5-4.5s). Singleton keyed by `os.path.realpath(file_path)`.
  `Article.text` is stripped at both ends. `Image:` links dropped, underscores in link targets become spaces.
- 254 chunks, 173 aliases, cold load ~0.6s, warm ~0.01s.
- `corpus.aliases` holds two kinds of entries: title variants, and bold terms mapped to the article that bolds them
  ("defined-in" aliases). Two that surprise: "gravity moss" -> Luminous Tree Frog, and
  "emotional weather broadcasting" -> Dr. Yuki Sasamoto. A2 deals with them.

## A2. Link resolution order (Window B)
`link_entity` answers "is there a page about this entity?" and must be precise. `recall` / `search_chunks` answer
"which pages talk about this text?" and must have recall. They do different jobs, so a None from `link_entity` is
information for the agent, never a dead end.

`link_entity(name)` returns the first hit in this order:
1. Title variants recomputed from `corpus.articles` (title; without leading honorific; without trailing " of <year>"),
   compared after stripping a plural "s"/"es" from each token, so "Gravitational Orchids" -> Gravitational Orchid.
2. Title-token containment (overlap >= 0.6 and >= 2 shared content tokens), same plural handling.
3. Bold-term aliases from `corpus.aliases`.

Title-based matches always beat bold aliases: "Emotional Weather Broadcasting" must resolve to the Service page.
Add `describe_link(name) -> tuple[str | None, str]` returning `(title, kind)` with kind in
`{"title", "partial", "bold", "none"}`; `link_entity(name)` is `describe_link(name)[0]`.

## A3. Agent loop: a failed link never ends the loop by itself (Window C)
For each candidate in `next`, in order:
1. `describe_link(name)` with kind title/partial -> add that article.
   kind bold -> add the article and put in FEEDBACK: "<name> has no article of its own; it is described in <title>."
2. kind none -> FEEDBACK "No article exists for <name>.", then add up to 2 whole articles from
   `retriever.recall(name, k=2)` that mention the name (every content token of the name occurs in the article) and are
   not in ctx yet.
3. If the whole pass over `next` added nothing and `gap` is non-empty, add the top article of `retriever.recall(gap, k=1)`
   that is not in ctx yet. This covers a `next` that is a description rather than a name.

The loop stops only when a full pass added nothing new, or after 4 rounds. Falling through to full-context does not
lose accuracy (all 28 pages are in that prompt) but costs ~50K tokens, so it has to be the exception.
`trace["mode"]` records which path produced the answer. Target on the dev set: at least 80% of multihop items answered
inside the agent loop.

## A4. has_support, precise version (Window C)
Normalise the answer and the verified quotes first: lowercase, remove thousands separators inside numbers, map the
number words zero..twenty to digits.

Split the answer's tokens into numeric tokens (contain a digit) and content tokens. Remove from the content tokens:
stopwords; honorifics (dr, sir, ms, mr, professor, captain); qualifiers (over, more, than, fewer, less, about,
approximately, around, nearly, almost, at, least, most, up, to, once, every); unit words (meter(s), kilometer(s), km,
year(s), day(s), hour(s), minute(s), percent, members, individuals, people, spectators).

- (a) at least one verified quote exists;
- (b) every numeric token occurs in the verified quotes;
- (c) one content token left -> it must occur in the verified quotes; two or more -> at least 60% must occur;
  none left ("3000", "over 500", "six days") -> skip (c), (b) decides.

An answer with neither numeric nor content tokens fails.

## A5. Rate limiting (Window C). Do this arithmetic before running 130 questions
- One full-context call is ~50K input tokens (~36K with `<ref>` bodies stripped). Five threads firing it together is
  250K tokens within seconds.
- An agent round is ~6K tokens (3 articles) growing to ~10K. A 3-hop question is ~25K over its rounds.
- Per question: ~10-25K in the loop, plus 50K whenever the fallback fires. If the fallback fires on a quarter of 130
  questions, that alone is ~1.6M tokens; the whole run is ~3-4M.
- With 5 threads the demand is roughly 0.5-1M tokens per minute. Any limit below that needs throttling.
- Size everything on raw input tokens. Do not assume prompt caching lowers rate-limit usage. Caching still helps cost
  and latency: confirm it with `usage.prompt_tokens_details.cached_tokens > 0` on the second full-context call.

Required:
1. At client init, read `x-ratelimit-limit-tokens` (TPM) from one tiny call. Print it.
2. One process-wide token bucket shared by all threads: capacity 0.8 x TPM, refilled continuously. Every LLM call
   reserves `len(prompt_chars) / 3.5 + 500` tokens first and blocks until they are available. The evaluator has no
   time limit: waiting is free, 429s and exceptions are not.
3. Mode by limit:
   - TPM < 60K: a full-context request cannot be sent at all. Disable it; the outcome is the agent loop's answer or IDK.
   - 60K <= TPM < 1M: full-context as fallback only, refs stripped, at most one full-context call in flight.
   - TPM >= 1M: as designed.
4. On a 429 anyway: honour `retry-after`, exponential backoff with jitter, up to 6 tries, then IDK.

## A6. Dev set (Window D)
- `src/dev_set.py` exists and passes its own check: 14 single, 8 multihop, 8 idk, 10 holdout. Items carry extra keys
  `tag` (finer category) and `trap` (the tempting wrong answer). Window D now only writes `src/run_dev.py`; break the
  results down by `type` and by `tag`.
- `src/dev_set_30.json` is a second, richer set: 9 categories, exact quotes, hop chains, shortcut checks, three
  150+ word questions. Support it with `--set json` if time allows. Its `difficulty` uses "mid" and its IDK answer is
  "I don't know".

## A7. Infobox and body disagree on four pages (Window C, prompt rule)
- Lunar Cartography Expedition of 1689: infobox says launch from Starrise, Yorkshire; body and the Aethermist Valley
  page say Aethermist Valley. The official hard example expects the valley.
- Emotional Weather Broadcasting Service: infobox founder is Dr. Empathia Feelwright; body and the Sasamoto page say it
  was founded by Dr. Yuki Sasamoto.
- Festival of Floating Lanterns: infobox location is Lumineshire; body says it takes place at the Levitating Library
  of Loftington.
- Academy of Forgotten Languages: infobox headquarters is Memoria; body and the Crysthaven page place it in the
  Crysthaven Islands (Prismheart).

Add to the system prompt: "If the infobox and the body text of an article disagree, prefer the version that another
article confirms; if no other article speaks to it, prefer the body text." A conflict is not a reason to abstain.

---

# Amendments v3 — ownership change, supersedes the "Architecture" section

`src/qa_agent.py` now contains a complete, self-contained `answer()` (corpus extraction, BM25 + embeddings +
RRF, planner loop of up to 4 hops, quote check + LLM entailment, full-context fallback, catch-all to IDK). It passes
both official examples and a 24-question set three runs in a row. **This file is the submission.**

- The session that wrote it owns `qa_agent.py` from now on. Nobody else edits it.
- The `agent.py` plan is dropped. Do not create it and do not rewire `qa_agent.py` onto `needle_index.py` /
  `retrieval.py`. Those two modules stay on disk only because `dev_set.py` uses `needle_index` for its own checks.
- `src/run_dev.py` exists. `src/dev_set.py` and `src/dev_set_30.json` are the dev sets.

Remaining work on `qa_agent.py`, in this order, each followed by `run_dev.py`:
1. An embeddings failure must not zero the run. At build time: on any exception, leave `chunk.vec` unset and run
   BM25-only. At query time: on any exception, rank by BM25 only for that query.
2. Rate limits: wrap `_chat_json` in an outer retry for RateLimitError / APITimeoutError / APIConnectionError with
   sleeps of 5, 15, 30, 60 seconds on top of the SDK retries. The evaluator has no time limit.
3. `{{coord|lat|lon|...}}` currently renders to nothing, so coordinates vanish from the evidence. Render it as "lat, lon".
4. Optional, keep only if the dev set does not get worse: the infobox/body conflict rule from A7.
