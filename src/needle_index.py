"""Needle index (CONTRACT.md, Window A): the 28 fictional "needle" pages of the enwiki dump.

Stdlib only, no network. The real pages are pure noise, so only the needles are indexed.

    corpus = load_corpus(file_path)    # thread-safe singleton per file
    corpus.articles[title]             # Article(title, text, links, timestamp)
    corpus.chunks                      # Chunk(id, title, section, text); chunks[i].id == i
    corpus.aliases[" ".join(tokens(name))] -> title
    corpus.squashed[title]             # squash(article.text), for verbatim-quote checks

Run `cd src && ../.venv/bin/python needle_index.py` for the self-test.
"""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass

IDK = "I don't know"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> list[str]:
    """Lowercase alphanumeric runs: "Dr. Yuki Sasamoto's" -> ["dr", "yuki", "sasamoto", "s"]."""
    return _TOKEN_RE.findall(s.lower())


def squash(s: str) -> str:
    """"".join(tokens(s)); punctuation/whitespace/case-insensitive form for verbatim-quote checks."""
    return "".join(tokens(s))


@dataclass(frozen=True)
class Article:
    title: str
    text: str  # full html-unescaped wikitext (infobox + body + refs), nothing removed
    links: tuple[str, ...]  # [[targets]] in order, deduped, no label/anchor, no Category:/File:
    timestamp: str


@dataclass(frozen=True)
class Chunk:
    id: int  # == index in Corpus.chunks
    title: str
    section: str  # "Infobox", "Lead", or the "== Section ==" heading
    text: str  # f"{title} > {section}\n" + verbatim section text


@dataclass
class Corpus:
    articles: dict[str, Article]  # title -> Article, sorted by title
    chunks: list[Chunk]
    aliases: dict[str, str]  # " ".join(tokens(alias)) -> title
    squashed: dict[str, str]  # title -> squash(article.text)


# --------------------------------------------------------------------------- extraction

NEEDLE_PREFIX = b'<page xmlns="http://www.mediawiki.org'
_PAGE_END = b"</page>"

_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
_TIMESTAMP_RE = re.compile(r"<timestamp>(.*?)</timestamp>", re.S)
_TEXT_RE = re.compile(r"<text\b[^>]*?(?:/>|>(.*?)</text>)", re.S)


def _parse_page(raw: str) -> dict | None:
    title = _TITLE_RE.search(raw)
    if not title:
        return None
    ts = _TIMESTAMP_RE.search(raw)
    text = _TEXT_RE.search(raw)
    return {
        "title": " ".join(html.unescape(title.group(1)).split()),
        "timestamp": ts.group(1).strip() if ts else "",
        "text": html.unescape(text.group(1) or "").strip() if text else "",
    }


_SCAN_BLOCK = 8 << 20


def _needle_offsets(f) -> list[int]:
    """Byte offsets of every needle prefix. Sequential block reads + bytes.find: ~0.6s for the 1 GB dump,
    where mmap.find takes 2.5-4.5s on macOS (page-fault bound)."""
    offsets = []
    keep = len(NEEDLE_PREFIX) - 1  # overlap so a prefix split across two blocks is found exactly once
    tail = b""
    base = 0  # file offset of buf[0]
    while True:
        block = f.read(_SCAN_BLOCK)
        if not block:
            return offsets
        buf = tail + block
        p = buf.find(NEEDLE_PREFIX)
        while p != -1:
            offsets.append(base + p)
            p = buf.find(NEEDLE_PREFIX, p + 1)
        tail = buf[-keep:]
        base += len(buf) - len(tail)


def _read_page(f, start: int) -> bytes | None:
    """Bytes from start through the next </page>; None if the page is truncated at EOF."""
    f.seek(start)
    data = b""
    while True:
        more = f.read(1 << 16)
        if not more:
            return None
        searched = max(0, len(data) - len(_PAGE_END) + 1)
        data += more
        end = data.find(_PAGE_END, searched)
        if end != -1:
            return data[:end + len(_PAGE_END)]


def _scan_needles(file_path: str) -> list[dict]:
    """Slice every needle page (prefix .. </page>) out of the dump and parse title/timestamp/text."""
    pages = []
    with open(file_path, "rb") as f:
        for start in _needle_offsets(f):
            raw = _read_page(f, start)
            if raw is None:
                continue
            page = _parse_page(raw.decode("utf-8", errors="replace"))
            if page:
                pages.append(page)
    return pages


# --------------------------------------------------------------------------- disk cache

_CACHE_VERSION = 1


def _cache_path(file_path: str) -> str:
    return file_path + ".needles.json"


def _file_sig(file_path: str) -> dict:
    st = os.stat(file_path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _read_cache(file_path: str, sig: dict) -> list[dict] | None:
    """Cached pages if the cache matches this file's size + mtime, else None. Never raises."""
    try:
        with open(_cache_path(file_path), encoding="utf-8") as f:
            data = json.load(f)
        if (data.get("version"), data.get("size"), data.get("mtime_ns")) != (
            _CACHE_VERSION, sig["size"], sig["mtime_ns"]):
            return None
        pages = data["pages"]
        if not pages or not all(
            isinstance(p, dict) and all(isinstance(p.get(k), str) for k in ("title", "timestamp", "text"))
            for p in pages
        ):
            return None
        return pages
    except Exception:
        return None


def _write_cache(file_path: str, sig: dict, pages: list[dict]) -> None:
    """Atomic write (tmp file + os.replace). Unwritable location or any other error: skip silently."""
    path = _cache_path(file_path)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path))
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": _CACHE_VERSION, **sig, "pages": pages}, f, ensure_ascii=False)
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


# --------------------------------------------------------------------------- parsing

_LINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
_SKIP_LINK_NS = ("category:", "file:", "image:")


def _extract_links(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out = []
    for m in _LINK_RE.finditer(text):
        target = m.group(1).split("|", 1)[0].split("#", 1)[0]
        target = " ".join(target.replace("_", " ").split()).lstrip(":").strip()
        if not target or target.lower().startswith(_SKIP_LINK_NS) or target in seen:
            continue
        seen.add(target)
        out.append(target)
    return tuple(out)


_HEADING_RE = re.compile(r"^[ \t]*(={2,6})[ \t]*(.+?)[ \t]*(={2,6})[ \t]*$", re.M)
_INFOBOX_RE = re.compile(r"\{\{\s*Infobox\b", re.I)
_BRACES_RE = re.compile(r"\{\{|\}\}")
_SKIP_SECTIONS = {"references", "external links", "see also"}


def _infobox_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the first {{Infobox ...}} block, brace-balanced; None if absent or unbalanced."""
    m = _INFOBOX_RE.search(text)
    if not m:
        return None
    depth = 0
    for b in _BRACES_RE.finditer(text, m.start()):
        depth += 1 if b.group() == "{{" else -1
        if depth == 0:
            return m.start(), b.end()
    return None


def _split_sections(text: str) -> list[tuple[str, str]]:
    """[(section, body)]: "Infobox", "Lead" (before the first ==), then each level-2 section with its
    subsections. Bodies are verbatim slices (infobox cut out of whichever region holds it)."""
    ib = _infobox_span(text)
    heads = []
    for m in _HEADING_RE.finditer(text):
        if min(len(m.group(1)), len(m.group(3))) != 2:
            continue
        if ib and ib[0] <= m.start() < ib[1]:
            continue
        heads.append((m.group(2).strip(" =\t"), m.start(), m.end()))

    regions = [("Lead", 0, heads[0][1] if heads else len(text))]
    for i, (name, _, body_start) in enumerate(heads):
        regions.append((name, body_start, heads[i + 1][1] if i + 1 < len(heads) else len(text)))

    out = []
    if ib:
        out.append(("Infobox", text[ib[0]:ib[1]]))
    for name, s, e in regions:
        if name.lower() in _SKIP_SECTIONS:
            continue
        if ib and ib[0] < e and ib[1] > s:
            body = text[s:max(s, ib[0])] + text[min(e, ib[1]):e]
        else:
            body = text[s:e]
        body = body.strip()
        if body:
            out.append((name, body))
    return out


_BOLD_RE = re.compile(r"'''(.+?)'''")
_PIPED_LINK_RE = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
_HONORIFICS = {"dr", "sir", "captain", "professor"}


def _bold_terms(text: str) -> list[str]:
    """Every '''bold term''', link markup reduced to its label, italics quotes removed."""
    out = []
    for m in _BOLD_RE.finditer(text):
        term = _PIPED_LINK_RE.sub(r"\1", m.group(1)).replace("''", "")
        out.append(" ".join(term.split()).strip(" '"))
    return out


def _title_variants(title: str) -> list[str]:
    """Alias keys derived from a title: without a leading honorific, without a trailing " of <year>", both."""
    toks = tokens(title)
    variants = []
    no_hon = toks[1:] if len(toks) > 1 and toks[0] in _HONORIFICS else toks
    for base in (toks, no_hon):
        variants.append(base)
        if len(base) > 2 and base[-2] == "of" and base[-1].isdigit():
            variants.append(base[:-2])
    return [" ".join(v) for v in variants if v]


def _build_aliases(articles: dict[str, Article]) -> dict[str, str]:
    """Priority: exact titles > title variants > '''bold terms''' (3-60 chars)."""
    aliases: dict[str, str] = {}
    for title in articles:
        aliases[" ".join(tokens(title))] = title
    for title in articles:
        for key in _title_variants(title):
            aliases.setdefault(key, title)
    for title, art in articles.items():
        for term in _bold_terms(art.text):
            key = " ".join(tokens(term))
            if key and 3 <= len(term) <= 60:
                aliases.setdefault(key, title)
    return aliases


def _assemble(pages: list[dict]) -> Corpus:
    by_title: dict[str, dict] = {}
    for p in pages:  # duplicate title: keep the latest revision
        old = by_title.get(p["title"])
        if old is None or p["timestamp"] >= old["timestamp"]:
            by_title[p["title"]] = p

    articles = {}
    for title in sorted(by_title):
        p = by_title[title]
        articles[title] = Article(title, p["text"], _extract_links(p["text"]), p["timestamp"])

    chunks: list[Chunk] = []
    for art in articles.values():
        for section, body in _split_sections(art.text):
            chunks.append(Chunk(len(chunks), art.title, section, f"{art.title} > {section}\n{body}"))

    return Corpus(
        articles=articles,
        chunks=chunks,
        aliases=_build_aliases(articles),
        squashed={t: squash(a.text) for t, a in articles.items()},
    )


# --------------------------------------------------------------------------- public entry point


def _build(file_path: str, read_cache: bool = True) -> Corpus:
    sig = _file_sig(file_path)
    pages = _read_cache(file_path, sig) if read_cache else None
    if pages is None:
        pages = _scan_needles(file_path)
        if not pages:
            # Not the expected dump. The fallback would be a full-corpus BM25 over all real pages;
            # out of scope per CONTRACT.md, so fail and let answer() return IDK.
            raise RuntimeError(f"no needle pages found in {file_path!r}")
        _write_cache(file_path, sig, pages)
    return _assemble(pages)


_LOCK = threading.Lock()
_CORPORA: dict[str, Corpus] = {}
_BUILD_COUNT = 0  # successful builds by load_corpus(); read by the self-test


def load_corpus(file_path: str) -> Corpus:
    """Thread-safe singleton per file (keyed by its real path). Raises if the file has no needles."""
    global _BUILD_COUNT
    key = os.path.realpath(file_path)
    corpus = _CORPORA.get(key)
    if corpus is None:
        with _LOCK:
            corpus = _CORPORA.get(key)
            if corpus is None:
                corpus = _build(file_path)
                _CORPORA[key] = corpus
                _BUILD_COUNT += 1
    return corpus


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    import sys

    data_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "enwiki", "enwiki-1gb.xml"
    )

    t0 = time.perf_counter()
    cold = _build(data_file, read_cache=False)
    t_cold = time.perf_counter() - t0
    cache_ok = _read_cache(data_file, _file_sig(data_file)) is not None
    t0 = time.perf_counter()
    warm = _build(data_file)
    t_warm = time.perf_counter() - t0

    print(f"articles: {len(cold.articles)}")
    for t, a in cold.articles.items():
        print(f"  {t}  ({len(a.text)} chars, {len(a.links)} links)")
    print(f"chunks: {len(cold.chunks)}")
    print(f"aliases: {len(cold.aliases)}")
    print(f"cold load (scan {os.path.getsize(data_file) / 1e9:.2f} GB): {t_cold:.3f}s")
    print(f"warm load ({'cache hit' if cache_ok else 'cache unavailable, rescanned'}): {t_warm:.3f}s")

    failures = []

    def check(ok, msg):
        if not ok:
            failures.append(msg)

    # Titles and exact-title needle->needle links as measured in CONTRACT.md "Data facts".
    expected_titles = {
        "Academy of Forgotten Languages", "Aethermist Valley", "Atmospheric Fishing League",
        "Competitive Cloud Shaping League", "Crystalline Butterfly Migration", "Crystalline Culture Movement",
        "Crysthaven Islands", "Culinary Time Travel Academy", "Dr. Amelia Thornwick", "Dr. Yuki Sasamoto",
        "Emotional Weather Broadcasting Service", "Festival of Floating Lanterns", "Gravitational Orchid",
        "Gravity Surfing Championship", "Great Harmonic Convergence of 1472", "Great Library Merger of Alexandria",
        "Holographic Memory Matrix", "Institute for Impossible Architecture", "Interdimensional Chess Tournament",
        "International Shadow Dance Championship", "Invisible Art Exhibition Society",
        "Levitating Library of Loftington", "Luminous Tree Frog", "Lunar Cartography Expedition of 1689",
        "Magnetophoric Moss", "Memory Foam Wrestling Federation", "Memory Metal Mining Consortium",
        "Meridian Dance Revolution",
    }
    abbrev = {
        "AFL": "Academy of Forgotten Languages", "AV": "Aethermist Valley", "CCM": "Crystalline Culture Movement",
        "CI": "Crysthaven Islands", "CTTA": "Culinary Time Travel Academy", "AT": "Dr. Amelia Thornwick",
        "YS": "Dr. Yuki Sasamoto", "EWBS": "Emotional Weather Broadcasting Service",
        "FFL": "Festival of Floating Lanterns", "GO": "Gravitational Orchid", "GSC": "Gravity Surfing Championship",
        "GLMA": "Great Library Merger of Alexandria", "HMM": "Holographic Memory Matrix",
        "IIA": "Institute for Impossible Architecture", "IAES": "Invisible Art Exhibition Society",
        "LLL": "Levitating Library of Loftington", "LTF": "Luminous Tree Frog",
        "LCE": "Lunar Cartography Expedition of 1689", "MM": "Magnetophoric Moss",
        "MMMC": "Memory Metal Mining Consortium", "MDR": "Meridian Dance Revolution",
    }
    expected_edges = {
        tuple(abbrev[x] for x in e.split(">"))
        for e in (
            "YS>AFL YS>EWBS LCE>AV MDR>CCM CCM>MDR GLMA>LLL LLL>GLMA AT>CI IIA>IAES IAES>IIA IIA>LLL "
            "LLL>FFL LLL>IIA GO>GSC GSC>GO GO>LLL CI>AFL CI>EWBS CI>MM LTF>AV AV>LTF AV>LCE EWBS>CI "
            "AFL>CI MMMC>FFL FFL>MMMC MMMC>HMM HMM>MMMC CTTA>MM MM>CTTA MM>CI FFL>LLL"
        ).split()
    }
    edges = {(t, d) for t, a in cold.articles.items() for d in a.links if d in cold.articles and d != t}

    check(len(cold.articles) == 28, f"expected 28 articles, got {len(cold.articles)}")
    check(set(cold.articles) == expected_titles,
          f"titles differ: missing {expected_titles - set(cold.articles)}, extra {set(cold.articles) - expected_titles}")
    check(edges == expected_edges,
          f"needle links differ: missing {expected_edges - edges}, extra {edges - expected_edges}")
    check(warm == cold, "warm (cached) corpus differs from cold corpus")

    for i, c in enumerate(cold.chunks):
        check(c.id == i, f"chunk {i} has id {c.id}")
        check(c.text.startswith(f"{c.title} > {c.section}\n"), f"chunk {i} header")
        check(c.section.lower() not in _SKIP_SECTIONS, f"chunk {i} is a skipped section")
    for t, a in cold.articles.items():
        secs = [c.section for c in cold.chunks if c.title == t]
        check("Lead" in secs, f"{t}: no Lead chunk")
        check(("Infobox" in secs) == (_infobox_span(a.text) is not None), f"{t}: Infobox chunk mismatch")
        check("&lt;" not in a.text and "&amp;" not in a.text, f"{t}: text not unescaped")
        check(cold.aliases.get(" ".join(tokens(t))) == t, f"{t}: title alias")
        check(cold.squashed[t] == squash(a.text), f"{t}: squashed")
        check(not any(l.lower().startswith(_SKIP_LINK_NS) or "|" in l or "#" in l for l in a.links),
              f"{t}: bad link")
        check(len(set(a.links)) == len(a.links), f"{t}: duplicate links")

    for alias, title in [
        ("Yuki Sasamoto", "Dr. Yuki Sasamoto"),
        ("Amelia Thornwick", "Dr. Amelia Thornwick"),
        ("Lunar Cartography Expedition", "Lunar Cartography Expedition of 1689"),
        ("Great Harmonic Convergence", "Great Harmonic Convergence of 1472"),
    ]:
        check(cold.aliases.get(" ".join(tokens(alias))) == title, f"alias {alias!r} -> {title!r}")
    for dangling in ("Bartholomew Moonwright", "Aurora Wolf", "Norathi people", "Mist Butterflies"):
        check(" ".join(tokens(dangling)) not in cold.aliases, f"dangling {dangling!r} must not be an alias")
    check("skyfathersbreath" in cold.squashed.get("Aethermist Valley", ""), "Aethermist Valley quote")
    check("Aethermist Valley" in cold.articles["Lunar Cartography Expedition of 1689"].links,
          "expedition -> valley link")

    # 5 threads hit load_corpus at once: exactly one build, everyone gets the same object.
    _CORPORA.clear()
    _BUILD_COUNT = 0
    barrier = threading.Barrier(5)
    results: list = [None] * 5

    def worker(i):
        barrier.wait()
        results[i] = load_corpus(data_file)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    check(_BUILD_COUNT == 1, f"load_corpus built {_BUILD_COUNT} times from 5 threads")
    check(all(r is results[0] for r in results), "threads got different Corpus objects")
    check(results[0] == cold, "threaded corpus differs from cold corpus")
    print(f"5-thread load_corpus: {_BUILD_COUNT} build, same object: {all(r is results[0] for r in results)}")

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("SELF-TEST PASSED")
