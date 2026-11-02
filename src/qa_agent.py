#!/usr/bin/env python3
"""
Multi-hop question answering over fictional articles hidden in a Wikipedia dump.

answer() runs a small retrieval-augmented pipeline over the fictional articles
hidden in the Wikipedia dump:

  1. Corpus: the fictional pages are the ones whose <page> tag carries an
     explicit MediaWiki xmlns attribute. They are cleaned, chunked by section,
     indexed with BM25 and embedded (embeddings are cached on disk; if
     embedding fails, ranking falls back to BM25 alone).
  2. Coarse recall: BM25 + vector search fused with reciprocal rank fusion.
  3. Multi-hop loop (max 4 hops): retrieve -> extract entities -> next hop.
  4. IDK signals (no recall / target field empty / dangling link) are
     collected as hints, never as an early return.
  5. Generate a short answer with citations.
  6. Check that the cited sources entail the answer.
  7. Fallback: re-ask once with every fictional page in context; if that
     answer is still unsupported, return "I don't know".
"""
import base64
import difflib
import hashlib
import html
import json
import math
import mmap
import os
import re
import sys
import threading
import time
from array import array
from collections import Counter
from dataclasses import dataclass

IDK = "I don't know"
PAGE_MARKER = b'<page xmlns="http://www.mediawiki.org/xml/export-0.11/"'
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "gpt-4.1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
DEBUG = bool(os.environ.get("ANSWER_DEBUG"))

MAX_HOPS = 4
TOP_K = 6
RRF_K = 60
CHUNK_CHARS = 1000
MIN_COSINE = 0.5  # best chunk cosine below this means "no recall" (BM25 always overlaps on common words)
BM25_K1, BM25_B = 1.5, 0.75
CHAT_RETRY_SLEEPS = (5, 15, 30, 60)  # outer retries on top of the SDK's own; no time limit

SKIP_SECTIONS = {"references", "external links", "see also", "further reading", "notes"}
HONORIFICS = ("dr. ", "sir ", "captain ", "professor ", "prof. ", "lady ", "lord ")
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
STOPWORDS = set("""
a an and are as at be been by did do does for from had has have how in is it its
of on or that the their them they this to was were what when where which who whom
whose why with would there these those than then into about after before during
""".split())


# ---------------------------------------------------------------------------
# Corpus: extraction, cleaning, chunking, indexing
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    title: str
    text: str
    tf: Counter
    length: int
    vec: array = None


@dataclass
class Corpus:
    titles: list
    chunks: list
    by_title: dict      # title -> [Chunk]
    aliases: dict       # lowercased name -> title
    link_targets: set   # lowercased wikilink targets found in any page
    idf: dict
    avgdl: float


def _extract_pages(file_path):
    """Return [(title, wikitext)] for every page tagged with PAGE_MARKER."""
    pages = []
    with open(file_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        pos = mm.find(PAGE_MARKER)
        while pos != -1:
            end = mm.find(b"</page>", pos)
            raw = mm[pos:end].decode("utf-8", errors="replace")
            title = re.search(r"<title>(.*?)</title>", raw, re.S)
            text = re.search(r"<text[^>]*>(.*?)</text>", raw, re.S)
            if title and text:
                pages.append((html.unescape(title.group(1)).strip(), html.unescape(text.group(1))))
            pos = mm.find(PAGE_MARKER, end)
    if not pages:
        raise RuntimeError(f"no fictional pages found in {file_path}")
    return pages


def _render_template(match):
    parts = [p.strip() for p in match.group(1).split("|")]
    name = parts[0].lower()
    args = [p for p in parts[1:] if "=" not in p]
    if name.startswith(("infobox", "speciesbox", "taxobox")):
        fields = []
        for part in parts[1:]:
            key, sep, value = part.partition("=")
            if sep and value.strip():
                fields.append(f"{key.strip().replace('_', ' ')}: {value.strip()}")
        return "\n\n" + "\n".join(fields) + "\n\n"
    if name == "short description" and args:
        return f"Short description: {args[0]}."
    if name == "coord" and len(args) >= 2 and all(re.fullmatch(r"-?\d+(\.\d+)?", a) for a in args[:2]):
        latlon = f"{args[0]}, {args[1]}"
        return f"Coordinates: {latlon}." if "display=title" in parts else latlon
    if name == "convert" and len(args) >= 2:
        return f"{args[0]} {args[1]}"
    if "date" in name and len(args) >= 3 and all(a.isdigit() for a in args[:3]):
        year, month, day = args[:3]
        if 1 <= int(month) <= 12:
            return f"{int(day)} {MONTHS[int(month) - 1]} {year}"
    if name in ("url", "official website") and args:
        return args[0]
    if name in ("nasdaq", "nyse") and args:
        return f"{name.upper()}: {args[0]}"
    return ""


def _clean_wikitext(raw):
    """Wikitext -> plain text. Returns (text, link_targets)."""
    text = "\n".join(line.strip() for line in raw.splitlines())
    links = {m.strip() for m in re.findall(r"\[\[([^\]|#]+)", text) if ":" not in m}
    text = re.sub(r"<ref\b[^>]*/>", "", text)
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:Category|File|Image):[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    while True:  # innermost templates first, so nested ones resolve
        rendered = re.sub(r"\{\{([^{}]*)\}\}", _render_template, text)
        if rendered == text:
            break
        text = rendered
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"<br\s*/?>", ", ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), links


def _split_sections(text):
    """Yield (section_name, body) pairs; level-3+ headings nest under level 2."""
    pieces = re.split(r"^(={2,})\s*(.*?)\s*\1\s*$", text, flags=re.M)
    yield "Introduction", pieces[0]
    parent = ""
    for i in range(1, len(pieces) - 2, 3):
        level, heading, body = len(pieces[i]), pieces[i + 1], pieces[i + 2]
        if level == 2:
            parent = heading
            name = heading
        else:
            name = f"{parent} / {heading}" if parent else heading
        if parent.lower() not in SKIP_SECTIONS:
            yield name, body


def _chunk_page(title, text):
    """Split a cleaned page into section-scoped chunks of ~CHUNK_CHARS."""
    chunks = []
    for section, body in _split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) > CHUNK_CHARS:
                chunks.append((section, buf))
                buf = ""
            buf = f"{buf}\n{para}" if buf else para
        if buf:
            chunks.append((section, buf))
    return [(f"{title}#{i}", f"{title} > {section}\n{body}") for i, (section, body) in enumerate(chunks)]


def _tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _aliases_for(title):
    name = title.lower()
    names = {name, re.sub(r"\s*\(.*?\)$", "", name)}
    for prefix in HONORIFICS:
        if name.startswith(prefix):
            names.add(name[len(prefix):])
    return names


def _build_corpus(file_path):
    titles, chunks, by_title, aliases, link_targets = [], [], {}, {}, set()
    for title, raw in _extract_pages(file_path):
        text, links = _clean_wikitext(raw)
        link_targets |= {link.lower() for link in links}
        titles.append(title)
        by_title[title] = []
        for alias in _aliases_for(title):
            aliases[alias] = title
        for chunk_id, chunk_text in _chunk_page(title, text):
            tokens = _tokenize(chunk_text)
            chunk = Chunk(chunk_id, title, chunk_text, Counter(tokens), len(tokens))
            chunks.append(chunk)
            by_title[title].append(chunk)

    df = Counter(t for c in chunks for t in c.tf)
    n = len(chunks)
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    avgdl = sum(c.length for c in chunks) / n

    try:
        vectors = _embed_with_cache([c.text for c in chunks], file_path)
    except Exception as e:  # embeddings are optional: rank with BM25 only
        print(f"[answer] embeddings unavailable ({_error_label(e)}); ranking with BM25 only",
              file=sys.stderr)
        vectors = [None] * len(chunks)
    for chunk, vec in zip(chunks, vectors):
        chunk.vec = vec
    return Corpus(titles, chunks, by_title, aliases, link_targets, idf, avgdl)


_CORPORA = {}
_CORPUS_LOCK = threading.Lock()


def _load_corpus(file_path):
    key = os.path.abspath(file_path)
    with _CORPUS_LOCK:
        if key not in _CORPORA:
            _CORPORA[key] = _build_corpus(key)
        return _CORPORA[key]


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _error_label(e):
    """Exception type plus HTTP status, without the message (it can echo the key)."""
    status = getattr(e, "status_code", None)
    return f"{type(e).__name__}" + (f" {status}" if status else "")


def _openai():
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            import openai
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            _CLIENT = openai.OpenAI(max_retries=5, timeout=120)
        return _CLIENT


def _chat_json(system, user):
    import openai
    transient = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    for pause in (*CHAT_RETRY_SLEEPS, None):
        try:
            response = _openai().chat.completions.create(
                model=ANSWER_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            break
        except transient as e:
            if pause is None:
                raise
            print(f"[answer] {_error_label(e)}; retrying in {pause}s", file=sys.stderr)
            time.sleep(pause)
    return json.loads(response.choices[0].message.content or "{}")


def _normalized(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array("f", (x / norm for x in vec))


def _embed(texts):
    vectors = []
    for i in range(0, len(texts), 100):
        response = _openai().embeddings.create(model=EMBED_MODEL, input=texts[i:i + 100])
        vectors.extend(_normalized(d.embedding) for d in response.data)
    return vectors


def _embed_with_cache(texts, file_path):
    """Embed texts, reusing vectors cached on disk by content hash."""
    cache_path = os.path.join(os.path.dirname(file_path), ".answer_cache", "embeddings.json")
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}

    keys = [hashlib.sha1(f"{EMBED_MODEL}\n{t}".encode()).hexdigest() for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        for i, vec in zip(missing, _embed([texts[i] for i in missing])):
            cache[keys[i]] = base64.b64encode(vec.tobytes()).decode()
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(cache, f)
            os.replace(tmp_path, cache_path)
        except OSError as e:
            print(f"[answer] could not write embedding cache: {e}", file=sys.stderr)

    vectors = []
    for k in keys:
        vec = array("f")
        vec.frombytes(base64.b64decode(cache[k]))
        vectors.append(vec)
    return vectors


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _bm25_scores(corpus, query):
    terms = set(_tokenize(query))
    scores = []
    for chunk in corpus.chunks:
        score = 0.0
        for t in terms:
            tf = chunk.tf.get(t)
            if tf:
                norm = 1 - BM25_B + BM25_B * chunk.length / corpus.avgdl
                score += corpus.idf[t] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
        scores.append(score)
    return scores


def _query_cosines(corpus, query):
    """Cosine of the query against every chunk, or None if vectors are unavailable."""
    if corpus.chunks[0].vec is None:
        return None
    try:
        qvec = _embed([query])[0]
    except Exception:
        return None
    return [sum(a * b for a, b in zip(qvec, c.vec)) for c in corpus.chunks]


def _hybrid_search(corpus, query, k=TOP_K):
    """BM25 + cosine rankings fused with RRF, or BM25 alone when the query can't
    be embedded. Returns (chunks, recalled, used_vectors)."""
    bm25 = _bm25_scores(corpus, query)
    rankings = [sorted((i for i, s in enumerate(bm25) if s > 0), key=lambda i: -bm25[i])]
    recalled = bool(rankings[0])
    cosine = _query_cosines(corpus, query)
    if cosine is not None:
        rankings.append(sorted(range(len(cosine)), key=lambda i: -cosine[i]))
        recalled = max(cosine) >= MIN_COSINE

    fused = Counter()
    for ranking in rankings:
        for rank, i in enumerate(ranking[:50]):
            fused[i] += 1 / (RRF_K + rank + 1)
    return [corpus.chunks[i] for i, _ in fused.most_common(k)], recalled, cosine is not None


def _resolve_entity(corpus, name):
    """Map an entity name to a page title, or None."""
    name = name.strip().lower()
    if name in corpus.aliases:
        return corpus.aliases[name]
    if len(name) < 5:
        return None
    candidates = [t for t in corpus.titles if name in t.lower() or t.lower() in name]
    return candidates[0] if len(candidates) == 1 else None


def _titles_in_question(corpus, question):
    q = question.lower()
    return {title for alias, title in corpus.aliases.items()
            if len(alias) >= 4 and re.search(rf"\b{re.escape(alias)}\b", q)}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You plan retrieval for a question-answering system over a small encyclopedia of fictional articles. Questions may need several hops (e.g. find where an event happened, then a species endemic to that place, then that species' population).

You get the question, the list of article titles, and the evidence retrieved so far. Work out what the evidence has established and what is still missing. Use only the evidence, never outside knowledge.

Return a JSON object:
{
  "target_field": "the exact fact the question asks for, e.g. 'current population estimate of <species>'",
  "known": "one or two sentences on what the evidence has established along the chain so far",
  "resolved": true only if the evidence explicitly states every link of the chain and the final answer (no link inferred from loose or indirect wording), else false,
  "next_entities": ["entities whose article should be read next; use the exact title from the list when one matches"],
  "next_query": "a short keyword search query for the next missing fact, or null if there is nothing left to look up",
  "field_missing": true if the article about the final target entity is in the evidence but does not state the requested fact, else false
}"""

ANSWER_SYSTEM = """You answer questions using only the evidence passages, which come from an encyclopedia of fictional articles. Never use outside knowledge: the articles are fictional and may contradict the real world.

Rules:
- Every entity and premise in the question must match the evidence. If the question asks about something the evidence does not mention, or its premise contradicts the evidence, answer "I don't know".
- If the requested fact is not explicitly stated for that exact entity, answer "I don't know". Do not guess, estimate, or borrow the fact from a similar entity.
- For multi-hop questions, follow the chain through the evidence and answer with the final fact.
- Give the shortest answer that fully answers the question (a name, number, date, place or short phrase), worded as in the source. No explanation.
- Retrieval signals, if given, are hints from the search step and can be false alarms; trust the evidence.

First write out the chain of evidence, then the answer. The chain has one step per link, starting from the entity the question names and ending with the passage that states the answer, so that the whole chain can be checked from the quotes alone. A single-hop question has one step. If the answer is "I don't know", the chain may be empty.

Return a JSON object:
{
  "chain": [
    {"fact": "one link, e.g. 'The expedition launched from Aethermist Valley'", "citation": "passage id", "quote": "the verbatim sentence from that passage stating this link"}
  ],
  "answer": "the answer, or I don't know"
}"""

ENTAIL_SYSTEM = """You verify answers produced by a question-answering system over fictional articles. Decide whether the source passages, read on their own, support the proposed answer to the question.

Supported: the passages explicitly state the answer for the exact entity and attribute the question asks about (following any chain of hops the question describes), and the question's premises match the passages. Minor wording or formatting differences are fine (e.g. "fewer than 3,000 individuals" supports "3000").
Unsupported: the answer is absent from the passages, taken from a different entity, contradicted, or depends on outside knowledge.

Return a JSON object: {"supported": true or false, "reason": "brief explanation"}"""


def _format_evidence(chunks):
    return "\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class _Trace:
    _print_lock = threading.Lock()

    def __init__(self, question):
        self.lines = [f"Q: {question}"]

    def log(self, msg):
        self.lines.append(f"  {msg}")

    def flush(self):
        if DEBUG:
            with self._print_lock:
                print("\n".join(self.lines), file=sys.stderr, flush=True)


def _multi_hop(question, corpus, trace):
    """Steps 2-4: coarse recall, hop loop, IDK signal collection."""
    evidence, signals = {}, []

    def add(chunks):
        for c in chunks:
            evidence.setdefault(c.id, c)

    for title in _titles_in_question(corpus, question):
        add(corpus.by_title[title])

    query, plan = question, {}
    for hop in range(MAX_HOPS):
        hits, recalled, used_vectors = _hybrid_search(corpus, query)
        add(hits)
        # Pages are small, so read the whole article behind the best hits:
        # bridging facts often sit in a neighbouring chunk that ranks lower.
        for title in dict.fromkeys(c.title for c in hits[:3]):
            add(corpus.by_title[title])
        if not recalled:
            signals.append(f"no_recall: nothing in the corpus matched the search '{query}'")
        trace.log(f"hop {hop}: query={query!r} hits={[c.id for c in hits]} recalled={recalled}"
                  + ("" if used_vectors else " (BM25 only)"))

        plan = _chat_json(
            PLANNER_SYSTEM,
            f"Question: {question}\n\nArticle titles: {'; '.join(corpus.titles)}\n\n"
            f"Evidence:\n{_format_evidence(evidence.values())}",
        )
        trace.log(f"  plan: {json.dumps(plan, ensure_ascii=False)}")

        new_pages = False
        for entity in plan.get("next_entities") or []:
            title = _resolve_entity(corpus, entity)
            if title:
                new_pages |= any(c.id not in evidence for c in corpus.by_title[title])
                add(corpus.by_title[title])
            else:
                linked = entity.strip().lower() in corpus.link_targets
                signals.append(f"dangling_link: '{entity}' has no article"
                               + (" although other articles link to it" if linked else ""))
        if plan.get("resolved"):
            break
        query = (plan.get("next_query") or "").strip()
        if not query and not new_pages:
            break
        query = query or question

    # Earlier hops often report the field missing before the right page is read,
    # so only the final plan's view counts.
    if plan.get("field_missing") and not plan.get("resolved"):
        signals.append(f"target_field_empty: '{plan.get('target_field')}' is not stated "
                       "in the article about the target entity")
    return list(evidence.values()), list(dict.fromkeys(signals))


def _generate(question, chunks, signals):
    """Step 5: answer + citations + supporting quotes."""
    user = f"Question: {question}\n\nEvidence:\n{_format_evidence(chunks)}"
    if signals:
        user += "\n\nRetrieval signals:\n" + "\n".join(f"- {s}" for s in signals)
    result = _chat_json(ANSWER_SYSTEM, user)
    chain = [s for s in result.get("chain") or [] if isinstance(s, dict)]
    return {
        "answer": str(result.get("answer") or IDK).strip(),
        "citations": [str(s.get("citation", "")) for s in chain],
        "quotes": [str(s.get("quote", "")) for s in chain if s.get("quote")],
    }


def _is_idk(text):
    t = text.strip().lower().rstrip(".")
    return not t or "don't know" in t or "do not know" in t or t in {"unknown", "n/a", "not stated"}


def _norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _quote_found(quote, sources):
    """True if every '...'-separated fragment of quote occurs in some source."""
    for fragment in re.split(r"\.\.\.|…", quote):
        fragment = _norm(fragment)
        if not fragment:
            continue
        for src in sources:
            if fragment in src:
                break
            match = difflib.SequenceMatcher(None, fragment, src, autojunk=False).find_longest_match(
                0, len(fragment), 0, len(src))
            if match.size >= 0.9 * len(fragment):
                break
        else:
            return False
    return True


def _entailed(question, result, corpus, trace, label):
    """Step 6: quotes must come from the evidence, then an LLM entailment check."""
    by_id = {c.id: c for c in corpus.chunks}
    cited = [by_id[i] for i in dict.fromkeys(result["citations"]) if i in by_id]
    if not cited or not result["quotes"]:
        trace.log(f"{label}: no valid citations/quotes")
        return False
    sources = [_norm(c.text) for c in cited]
    missing = [q for q in result["quotes"] if not _quote_found(q, sources)]
    if missing:
        trace.log(f"{label}: quotes not found in cited passages: {missing}")
        return False
    verdict = _chat_json(
        ENTAIL_SYSTEM,
        f"Question: {question}\nProposed answer: {result['answer']}\n\n"
        f"Source passages:\n{_format_evidence(cited)}",
    )
    trace.log(f"{label}: entailment={json.dumps(verdict, ensure_ascii=False)}")
    return bool(verdict.get("supported"))


def answer(question: str, file_path: str) -> str:
    """
    Answer questions based on information in the Wikipedia articles file.

    Args:
        question: The question to answer
        file_path: Path to the file containing Wikipedia articles

    Returns:
        The answer to the question, or "I don't know" if unanswerable
    """
    trace = _Trace(question)
    try:
        corpus = _load_corpus(file_path)
        evidence, signals = _multi_hop(question, corpus, trace)
        trace.log(f"signals: {signals}")

        result = _generate(question, evidence, signals)
        trace.log(f"rag answer: {result}")
        if not _is_idk(result["answer"]) and _entailed(question, result, corpus, trace, "rag"):
            return result["answer"]

        result = _generate(question, corpus.chunks, signals)
        trace.log(f"full-context answer: {result}")
        if not _is_idk(result["answer"]) and _entailed(question, result, corpus, trace, "full-context"):
            return result["answer"]
        return IDK
    except Exception as e:
        trace.log(f"error: {type(e).__name__}: {e}")
        return IDK
    finally:
        trace.flush()


# Example questions to help you develop and test your solution.
# The graded evaluation uses a separate encrypted set of questions -- these
# two examples are illustrative only and will NOT appear in that set.
EXAMPLES = [
    {
        "question": "What do the Norathi people call the Aethermist Valley?",
        "answer": "Skyfather's Breath",
        "difficulty": "easy",
    },
    {
        "question": (
            "What is the current population estimate of the species that is "
            "endemic to the valley from which the Lunar Cartography Expedition "
            "was launched?"
        ),
        "answer": "3000",
        "difficulty": "hard",
    },
]


if __name__ == "__main__":
    import os
    import sys

    from evaluator import Evaluator

    data_file = os.path.join(
        os.path.dirname(__file__), "..", "data", "enwiki", "enwiki-1gb.xml"
    )

    if not os.path.exists(data_file):
        print(f"Error: Wikipedia XML file not found at {data_file}")
        print("Run `python setup.py` from the repository root first.")
        sys.exit(1)

    evaluator = Evaluator(EXAMPLES, verbose=True)
    evaluator.run_evaluation(answer, data_file, max_workers=5)
