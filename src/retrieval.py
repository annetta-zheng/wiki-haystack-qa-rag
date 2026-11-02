"""Retrieval (CONTRACT.md, Window B): entity linking and article/chunk recall over the needle corpus.

    retr = get_retriever(file_path, client)      # thread-safe singleton per file
    retr.link_entity("the Aethermist Valley")    # -> "Aethermist Valley"; dangling names -> None
    retr.describe_link("Gravity Moss")           # -> ("Luminous Tree Frog", "bold"): kind title|partial|bold|none
    retr.recall(question, k=3)                   # article titles, best first (alias hits first)
    retr.search_chunks(name, k=6, exclude_titles=ctx)   # Chunk objects, best first

Ranking: alias hits in the query, then BM25 over chunks fused (RRF) with an optional embedding ranking.
The Retriever is read-only after construction. The vector path is optional: any API failure, at build or
query time, switches it off for the whole process and BM25 carries on alone.

Run `cd src && ../.venv/bin/python retrieval.py` for the self-test.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import re
import tempfile
import threading
from collections import defaultdict

from needle_index import Chunk, Corpus, load_corpus, tokens

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or other
our ours out over own same she should so some such than that the their theirs them then there these they this
those through to too under until up very was we were what when where which while who whom whose why will with
would you your
""".split())
HONORIFICS = frozenset({"dr", "doctor", "sir", "captain", "professor", "prof", "mr", "mrs", "ms"})
_ARTICLES = frozenset({"the", "a", "an"})

BM25_K1 = 1.5
BM25_B = 0.75
TITLE_WEIGHT = 3
TOP_N = 20  # per-ranker depth before fusion
RRF_K = 60

LINK_MIN_RATIO = 0.6
LINK_MIN_SHARED = 2

EMB_MODEL = "text-embedding-3-small"
EMB_DIMS = 256
EMB_BATCH = 100  # inputs per embeddings request
EMB_CACHE_VERSION = 1
# Embeddings get a shorter leash than the shared client's (timeout=60, max_retries=5): a slow or failing
# endpoint costs one quick failure, then the process is BM25-only.
BUILD_OPTS = {"timeout": 30, "max_retries": 2}
QUERY_OPTS = {"timeout": 10, "max_retries": 2}


def _stem(t: str) -> str:
    """Plural "s"/"es" -> singular, so both forms meet: orchids -> orchid, butterflies -> butterfly,
    churches -> church, languages -> language. Leaves moss, nexus, atlantis alone. Applied identically
    to queries, chunks and titles."""
    if len(t) <= 3 or not t.endswith("s") or t.endswith(("ss", "us", "is")):
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("ches", "shes", "sses", "xes", "zes")):
        return t[:-2]
    return t[:-1]


def _plural_key(toks) -> tuple[str, ...]:
    return tuple(_stem(t) for t in toks)


def _title_variants(title: str) -> list[list[str]]:
    """Token lists for a title: as is, without a leading honorific, without a trailing " of <year>", both."""
    toks = tokens(title)
    out = []
    for base in (toks, toks[1:] if len(toks) > 1 and toks[0] in HONORIFICS else toks):
        out.append(base)
        if len(base) > 2 and base[-2] == "of" and base[-1].isdigit():
            out.append(base[:-2])
    return [v for v in out if v]


def _content(toks) -> set[str]:
    """Content tokens: no stopwords, honorifics or single characters (possessive "s"), stemmed."""
    return {_stem(t) for t in toks if len(t) > 1 and t not in STOPWORDS and t not in HONORIFICS}


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in tokens(text) if t not in STOPWORDS]


# ---------------------------------------------------------------------- vector path

_VEC_LOCK = threading.Lock()
_vector_error: str | None = None  # first failure; once set, the vector path is off for the process
_KEY_RE = re.compile(r"sk-[A-Za-z0-9_*\-]+")


def vector_error() -> str | None:
    """Why the vector path is off for this process, or None if it has not failed."""
    return _vector_error


def _disable_vector(exc: BaseException) -> None:
    global _vector_error
    status = getattr(exc, "status_code", None)
    msg = _KEY_RE.sub("sk-***", " ".join(str(exc).split()))[:300]
    with _VEC_LOCK:
        if _vector_error is None:
            _vector_error = f"{type(exc).__name__}{f' {status}' if status else ''}: {msg}"


def _with_options(client, opts: dict):
    """Copy of the client with other timeout/retries; shares its connection pool."""
    try:
        return client.with_options(**opts)
    except Exception:
        return client


def _normalize(v) -> tuple[float, ...]:
    n = math.sqrt(sum(x * x for x in v))
    if not n:
        raise ValueError("zero-length embedding")
    return tuple(x / n for x in v)


def _embed(client, texts: list[str]) -> list[tuple[float, ...]]:
    """Unit-length embeddings, one per text, in input order. Raises on any malformed response."""
    resp = client.embeddings.create(model=EMB_MODEL, input=texts, dimensions=EMB_DIMS)
    data = sorted(resp.data, key=lambda d: d.index)
    if len(data) != len(texts) or any(len(d.embedding) != EMB_DIMS for d in data):
        raise ValueError(f"embeddings response: {len(data)} vectors for {len(texts)} inputs")
    return [_normalize(d.embedding) for d in data]


def _emb_key(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps([EMB_MODEL, EMB_DIMS, texts]).encode("utf-8")).hexdigest()


def _read_emb_cache(path: str, key: str, n: int) -> list[tuple[float, ...]] | None:
    """Cached vectors if the cache was built from exactly these chunk texts, else None. Never raises."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if (data.get("version"), data.get("key")) != (EMB_CACHE_VERSION, key):
            return None
        vecs = data["vectors"]
        if len(vecs) != n or any(len(v) != EMB_DIMS for v in vecs):
            return None
        return [_normalize([float(x) for x in v]) for v in vecs]
    except Exception:
        return None


def _write_emb_cache(path: str, key: str, vecs: list[tuple[float, ...]]) -> None:
    """Atomic write (tmp file + os.replace). Unwritable location or any other error: skip silently."""
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path))
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": EMB_CACHE_VERSION, "key": key, "model": EMB_MODEL, "dims": EMB_DIMS,
                       "vectors": [[round(x, 6) for x in v] for v in vecs]}, f, separators=(",", ":"))
        os.chmod(tmp, 0o644)  # mkstemp creates 0600
        os.replace(tmp, path)
        tmp = None
    except Exception:
        pass
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


class Retriever:
    def __init__(self, corpus: Corpus, client=None, *, cache_path: str | None = None):
        """client: openai.OpenAI or None (BM25 only). cache_path: optional embeddings cache file;
        get_retriever passes file_path + ".emb.json"."""
        self.corpus = corpus
        self.chunks: list[Chunk] = corpus.chunks
        self.client = client

        # -- describe_link: title variants recomputed from the titles (plural-insensitive keys), and the
        #    content-token sets of each variant for containment. Full titles first, so a title never loses
        #    to another article's variant.
        self._title_keys: dict[tuple[str, ...], str] = {_plural_key(tokens(t)): t for t in corpus.articles}
        self._title_sets: dict[str, list[set[str]]] = {}
        for title in corpus.articles:
            sets: list[set[str]] = []
            for v in _title_variants(title):
                self._title_keys.setdefault(_plural_key(v), title)
                if (c := _content(v)) and c not in sets:
                    sets.append(c)
            self._title_sets[title] = sets

        # -- recall step (1): alias n-grams
        self._max_alias_len = max((len(a.split()) for a in corpus.aliases), default=0)

        # -- BM25 over chunks; the title field counts TITLE_WEIGHT times
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doclen: list[int] = []
        for c in self.chunks:
            header, _, body = c.text.partition("\n")
            terms = _terms(c.title) * TITLE_WEIGHT + _terms(c.section) + _terms(body)
            tf: dict[str, int] = defaultdict(int)
            for t in terms:
                tf[t] += 1
            for t, n in tf.items():
                postings[t].append((c.id, n))
            self._doclen.append(len(terms))
        n_docs = len(self.chunks)
        self._avgdl = (sum(self._doclen) / n_docs) if n_docs else 1.0
        self._postings = dict(postings)
        self._idf = {t: math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}

        # -- optional vector path: every chunk embedded once (or loaded from cache_path)
        self._vecs: list[tuple[float, ...]] | None = None
        self._query_client = None
        if client is not None and _vector_error is None:
            try:
                self._vecs = self._chunk_vectors(client, cache_path)
                self._query_client = _with_options(client, QUERY_OPTS)
            except Exception as e:
                self._vecs = None
                _disable_vector(e)

    def _chunk_vectors(self, client, cache_path: str | None) -> list[tuple[float, ...]]:
        texts = [c.text for c in self.chunks]
        key = _emb_key(texts)
        if cache_path:
            cached = _read_emb_cache(cache_path, key, len(texts))
            if cached is not None:
                return cached
        build_client = _with_options(client, BUILD_OPTS)
        vecs: list[tuple[float, ...]] = []
        for i in range(0, len(texts), EMB_BATCH):
            vecs.extend(_embed(build_client, texts[i:i + EMB_BATCH]))
        if cache_path:
            _write_emb_cache(cache_path, key, vecs)
        return vecs

    @property
    def vector_enabled(self) -> bool:
        return self._vecs is not None and _vector_error is None

    # ------------------------------------------------------------------ entity linking

    def describe_link(self, name: str) -> tuple[str | None, str]:
        """(title, kind): does this entity have a page? First hit wins, titles always beat bold aliases:
          "title"   - a title variant (title; no leading honorific; no trailing " of <year>"), plural-insensitive
          "partial" - title-token containment: shared / max(len) >= 0.6 both ways, >= 2 shared content tokens
          "bold"    - a bold term of the article that describes it; the entity has no page of its own
          "none"    - (None, "none"). Precision over recall: dangling entities land here."""
        toks = tokens(name or "")
        if not toks:
            return None, "none"
        # the name as given, and without a possessive 's and leading articles/honorifics
        stripped = list(toks)
        if len(stripped) > 1 and stripped[-1] == "s":
            stripped.pop()
        while len(stripped) > 1 and (stripped[0] in _ARTICLES or stripped[0] in HONORIFICS):
            stripped.pop(0)
        forms = (toks, stripped) if stripped != toks else (toks,)

        for form in forms:
            title = self._title_keys.get(_plural_key(form))
            if title is not None:
                return title, "title"
        title = self._containment(toks)
        if title is not None:
            return title, "partial"
        for form in forms:
            title = self.corpus.aliases.get(" ".join(form))
            if title is not None:
                return title, "bold"
        return None, "none"

    def link_entity(self, name: str) -> str | None:
        """Article title for an entity name, or None (= no page of its own and not a bold term anywhere)."""
        return self.describe_link(name)[0]

    def _containment(self, toks: list[str]) -> str | None:
        """Best title by content-token overlap; None below the thresholds or on a tie between titles."""
        want = _content(toks)
        if len(want) < LINK_MIN_SHARED:
            return None
        best, best_score, tied = None, None, False
        for title, sets in self._title_sets.items():
            score = None
            for have in sets:
                shared = len(want & have)
                ratio = shared / max(len(want), len(have))
                if shared >= LINK_MIN_SHARED and ratio >= LINK_MIN_RATIO:
                    score = max(score or (0.0, 0), (ratio, shared))
            if score is None:
                continue
            if best_score is None or score > best_score:
                best, best_score, tied = title, score, False
            elif score == best_score:
                tied = True
        return None if tied else best

    # ------------------------------------------------------------------ recall

    def _alias_hits(self, query: str) -> list[str]:
        """Articles whose alias occurs as a contiguous token run of the query, longest alias first.
        A match lying inside a longer accepted match is dropped ("crystal palace" inside
        "crystal palace of consciousness")."""
        q = tokens(query)
        aliases = self.corpus.aliases
        covered = [False] * len(q)
        hits: list[str] = []
        for n in range(min(self._max_alias_len, len(q)), 0, -1):
            for i in range(len(q) - n + 1):
                title = aliases.get(" ".join(q[i:i + n]))
                if title is None or all(covered[i:i + n]):
                    continue
                for j in range(i, i + n):
                    covered[j] = True
                if title not in hits:
                    hits.append(title)
        return hits

    def _bm25(self, query: str, exclude: frozenset[str] = frozenset()) -> list[tuple[int, float]]:
        """[(chunk id, score)] for chunks matching at least one query term, best first."""
        scores: dict[int, float] = defaultdict(float)
        for t in set(_terms(query)):
            idf = self._idf.get(t)
            if idf is None:
                continue
            for cid, tf in self._postings[t]:
                norm = BM25_K1 * (1 - BM25_B + BM25_B * self._doclen[cid] / self._avgdl)
                scores[cid] += idf * tf * (BM25_K1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if exclude:
            ranked = [(cid, s) for cid, s in ranked if self.chunks[cid].title not in exclude]
        return ranked

    def _vector(self, query: str, exclude: frozenset[str] = frozenset()) -> list[int]:
        """Chunk ids by cosine similarity to the query, best first; [] when the vector path is off."""
        if not self.vector_enabled or not query.strip():
            return []
        try:
            (q,) = _embed(self._query_client, [query])
        except Exception as e:
            _disable_vector(e)
            return []
        sims = [(sum(map(operator.mul, q, v)), cid) for cid, v in enumerate(self._vecs)
                if not exclude or self.chunks[cid].title not in exclude]
        sims.sort(key=lambda sc: (-sc[0], sc[1]))
        return [cid for _, cid in sims]

    def _fused(self, query: str, exclude: frozenset[str] = frozenset()) -> list[int]:
        """Chunk ids ranked by RRF over the BM25 top 20 and the vector top 20 (when available)."""
        bm = self._bm25(query, exclude)[:TOP_N]
        bm_score = dict(bm)
        rankings = [[cid for cid, _ in bm], self._vector(query, exclude)[:TOP_N]]
        rrf: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, cid in enumerate(ranking, 1):
                rrf[cid] += 1.0 / (RRF_K + rank)
        return sorted(rrf, key=lambda cid: (-rrf[cid], -bm_score.get(cid, 0.0), cid))

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Up to k article titles, best first: alias hits (longest first), then articles by best fused chunk."""
        if k <= 0:
            return []
        out = self._alias_hits(query or "")
        for cid in self._fused(query or ""):
            title = self.chunks[cid].title
            if title not in out:
                out.append(title)
        return out[:k]

    def search_chunks(self, query: str, k: int = 6, exclude_titles=()) -> list[Chunk]:
        """Top k chunks for the query by fused rank, skipping chunks of the excluded articles."""
        if k <= 0:
            return []
        exclude = frozenset(exclude_titles or ())
        return [self.chunks[cid] for cid in self._fused(query or "", exclude)[:k]]


# ---------------------------------------------------------------------- singleton

_LOCK = threading.Lock()
_RETRIEVERS: dict[str, Retriever] = {}
_BUILD_COUNT = 0  # successful builds by get_retriever(); read by the self-test


def get_retriever(file_path: str, client=None) -> Retriever:
    """Thread-safe singleton per file (keyed by its real path). Raises only if load_corpus raises."""
    global _BUILD_COUNT
    key = os.path.realpath(file_path)
    retr = _RETRIEVERS.get(key)
    if retr is None:
        with _LOCK:
            retr = _RETRIEVERS.get(key)
            if retr is None:
                retr = Retriever(load_corpus(file_path), client, cache_path=file_path + ".emb.json")
                _RETRIEVERS[key] = retr
                _BUILD_COUNT += 1
    return retr


# ---------------------------------------------------------------------- self-test

if __name__ == "__main__":
    import sys
    import time

    data_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "enwiki", "enwiki-1gb.xml"
    )
    failures = []

    def check(ok, msg):
        if not ok:
            failures.append(msg)
        return ok

    corpus = load_corpus(data_file)
    t0 = time.perf_counter()
    bm25 = Retriever(corpus, None)
    print(f"BM25 index: {len(bm25.chunks)} chunks, {len(bm25._postings)} terms, "
          f"built in {time.perf_counter() - t0:.3f}s")

    examples = [
        ("What do the Norathi people call the Aethermist Valley?", "Aethermist Valley"),
        ("What is the current population estimate of the species that is endemic to the valley from which the "
         "Lunar Cartography Expedition was launched?", "Lunar Cartography Expedition of 1689"),
    ]

    def show_recall(retr, label):
        print(f"\nrecall() [{label}]")
        for q, first_hop in examples:
            got = retr.recall(q, k=3)
            fused = []
            for cid in retr._fused(q):
                if retr.chunks[cid].title not in fused:
                    fused.append(retr.chunks[cid].title)
            ok = check(got[:1] == [first_hop], f"[{label}] recall({q[:40]!r}...) rank 1 = {got[:1]}, want {first_hop!r}")
            print(f"  Q: {q}")
            print(f"     recall(k=3)     = {got}   rank-1 {'OK' if ok else 'FAIL'}")
            print(f"     alias hits      = {retr._alias_hits(q)}")
            print(f"     fused (no alias) = {fused[:5]}")

    show_recall(bm25, "BM25 only")

    print("\ndescribe_link() / link_entity()")
    EWBS, LCE = "Emotional Weather Broadcasting Service", "Lunar Cartography Expedition of 1689"
    link_cases = [
        # required by CONTRACT.md
        ("Bartholomew Moonwright", None, "none"), ("Aurora Wolf", None, "none"), ("Norathi people", None, "none"),
        ("Yuki Sasamoto", "Dr. Yuki Sasamoto", "title"), ("Lunar Cartography Expedition", LCE, "title"),
        ("the Aethermist Valley", "Aethermist Valley", "title"),
        # A2: plural handling, and titles beat bold aliases
        ("Gravitational Orchids", "Gravitational Orchid", "title"),
        ("Emotional Weather Broadcasting", EWBS, "partial"),
        ("Gravity Moss", "Luminous Tree Frog", "bold"), ("Khystari people", "Crysthaven Islands", "bold"),
        ("Academy of Forgotten Language", "Academy of Forgotten Languages", "title"),
        ("Crysthaven Island", "Crysthaven Islands", "title"), ("Magnetophoric Mosses", "Magnetophoric Moss", "title"),
        ("Floating Lanterns Festival", "Festival of Floating Lanterns", "partial"),
        # more precision / normalisation checks
        ("Sir Bartholomew Moonwright", None, "none"), ("Mist Butterflies", None, "none"),
        ("Cosmos Wavewright", None, "none"), ("Library of Alexandria", None, "none"), ("Time travel", None, "none"),
        ("", None, "none"), ("Dr. Yuki Sasamoto's", "Dr. Yuki Sasamoto", "title"),
        ("the Great Harmonic Convergence", "Great Harmonic Convergence of 1472", "title"),
        ("Shadow Dance Championship", "International Shadow Dance Championship", "partial"),
    ]
    for name, want, kind in link_cases:
        got = bm25.describe_link(name)
        ok = check(got == (want, kind) and bm25.link_entity(name) == want,
                   f"describe_link({name!r}) = {got!r}, want {(want, kind)!r}")
        print(f"  {name!r:34} -> {got[0]!r:45} {got[1]:8} {'OK' if ok else 'FAIL'}")

    # Informational (A2): bold aliases that a title match now overrides.
    title_keys = {" ".join(v) for t in corpus.articles for v in _title_variants(t)}
    overridden = [(k, t, bm25.describe_link(k)) for k, t in sorted(corpus.aliases.items())
                  if k not in title_keys and bm25.describe_link(k) != (t, "bold")]
    print(f"\nbold aliases overridden by a title match: {len(overridden)}")
    for k, t, (got, kind) in overridden:
        print(f"  {k!r:40} bold -> {t!r}, now {kind} -> {got!r}")

    # Informational: how the dangling [[links]] of the needles resolve (none of them has its own page).
    by_kind: dict[str, list[str]] = defaultdict(list)
    for art in corpus.articles.values():
        for target in art.links:
            if target not in corpus.articles:
                got, kind = bm25.describe_link(target)
                if got is not None and target not in by_kind[kind]:
                    by_kind[kind].append(target)
                    if kind != "bold":
                        print(f"  dangling {target!r:30} -> {got!r}  [{kind}]")
    print(f"dangling link targets resolved: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_kind.items())))

    print("\nsearch_chunks()")
    for q, excl in (("Aurora Wolf", ()), ("Bartholomew Moonwright", ("Lunar Cartography Expedition of 1689",))):
        got = bm25.search_chunks(q, k=3, exclude_titles=excl)
        check(all(c.title not in excl for c in got), f"search_chunks({q!r}) returned an excluded title")
        print(f"  {q!r} exclude={list(excl)}: {[f'{c.title} > {c.section}' for c in got]}")
    check(bm25.recall("", k=3) == [] and bm25.search_chunks("", k=3) == [], "empty query must return []")

    # ---- vector path (needs OPENAI_API_KEY in src/.env; skipped otherwise)
    client = None
    try:
        import openai
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        if os.environ.get("OPENAI_API_KEY"):
            client = openai.OpenAI(timeout=60, max_retries=5)
    except Exception as e:
        print(f"\nvector: no client ({type(e).__name__})")
    emb_cache = data_file + ".emb.json"

    if client is None:
        print("\nvector: skipped (no OPENAI_API_KEY)")
    else:
        t0 = time.perf_counter()
        cold = Retriever(corpus, client)  # no cache: embeds every chunk
        t_cold = time.perf_counter() - t0
        print(f"\nvector build, no cache: {t_cold:.2f}s, enabled={cold.vector_enabled}, error={vector_error()}")
        if cold.vector_enabled:
            n_req = -(-len(corpus.chunks) // EMB_BATCH)
            print(f"embeddings endpoint: OK ({len(corpus.chunks)} chunks, {n_req} requests, {EMB_DIMS} dims)")
            Retriever(corpus, client, cache_path=emb_cache)  # writes the cache if it is missing or stale
            t0 = time.perf_counter()
            warm = Retriever(corpus, client, cache_path=emb_cache)
            t_warm = time.perf_counter() - t0
            hit = _read_emb_cache(emb_cache, _emb_key([c.text for c in corpus.chunks]), len(corpus.chunks))
            print(f"vector build, cache {'hit' if hit else 'unavailable'}: {t_warm:.3f}s")
            check(warm.vector_enabled, "warm retriever lost its vectors")
            show_recall(warm, "BM25 + vector")
            for q, _ in examples:
                top = [f"{warm.chunks[c].title} > {warm.chunks[c].section}" for c in warm._vector(q)[:3]]
                print(f"  vector top 3 for {q[:50]!r}...: {top}")
            print("\nsearch_chunks() [BM25 + vector]")
            for q in ("Aurora Wolf", "who led the moon expedition"):
                print(f"  {q!r}: {[f'{c.title} > {c.section}' for c in warm.search_chunks(q, k=3)]}")
        else:
            print(f"embeddings endpoint: FAILED -> BM25 only ({vector_error()})")

    # 5 threads hit get_retriever at once (with the real client, if any): one build, same object.
    _RETRIEVERS.clear()
    _BUILD_COUNT = 0
    barrier = threading.Barrier(5)
    results: list = [None] * 5

    def worker(i):
        barrier.wait()
        results[i] = get_retriever(data_file, client)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    same = all(r is results[0] for r in results)
    check(_BUILD_COUNT == 1, f"get_retriever built {_BUILD_COUNT} times from 5 threads")
    check(same, "threads got different Retriever objects")
    print(f"\n5-thread get_retriever: {_BUILD_COUNT} build, same object: {same}, "
          f"vector_enabled={results[0].vector_enabled}")

    # ---- degradation: these flip the process-wide "vector off" flag, so they run last.
    # An unreachable endpoint (nothing listens on port 9) stands in for a 403 / timeout / outage.
    try:
        import openai

        dead = openai.OpenAI(api_key="sk-selftest-invalid", base_url="http://127.0.0.1:9/v1", timeout=2,
                             max_retries=0)
    except Exception:
        dead = None
    if dead is not None:
        print("\ndegradation:")
        want = {q: bm25.recall(q, k=3) for q, _ in examples}
        cached = _read_emb_cache(emb_cache, _emb_key([c.text for c in corpus.chunks]), len(corpus.chunks))
        if cached is not None and _vector_error is None:
            r = Retriever(corpus, dead, cache_path=emb_cache)  # vectors from the cache, no request at build
            check(r.vector_enabled, "cache hit should leave the vector path on until the first query")
            got = {q: r.recall(q, k=3) for q, _ in examples}
            check(not r.vector_enabled and got == want, "query-time failure must fall back to BM25")
            print(f"  query-time failure: vector_enabled -> {r.vector_enabled}, recall == BM25 only: {got == want}")
            print(f"    error: {vector_error()}")
        _vector_error = None  # self-test only: re-arm so the build-time failure is exercised too
        t0 = time.perf_counter()
        r = Retriever(corpus, dead)  # no cache: the first embeddings request fails during build
        got = {q: r.recall(q, k=3) for q, _ in examples}
        check(not r.vector_enabled and got == want, "build-time failure must fall back to BM25")
        print(f"  build-time failure ({time.perf_counter() - t0:.1f}s): vector_enabled={r.vector_enabled}, "
              f"recall == BM25 only: {got == want}")
        print(f"    error: {vector_error()}")

    if failures:
        print("\nSELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nSELF-TEST PASSED")
