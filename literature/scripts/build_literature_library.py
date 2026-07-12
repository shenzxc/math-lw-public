#!/usr/bin/env python3
"""Build a local literature library for the UCKV paper project.

The script uses public metadata and open-access PDF URLs first:
OpenAlex, arXiv, OpenReview, ACL Anthology, PMLR, and publisher OA links.
It does not use credentials, bypass paywalls, or scrape restricted PDFs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LIT = ROOT / "literature"
PDF_DIR = LIT / "pdfs"
META_DIR = LIT / "metadata"
REPORT_DIR = LIT / "reports"
SCOPUS_CANDIDATES = Path("/tmp/scopus_kv_top100.json")

OPENALEX = "https://api.openalex.org/works"
ARXIV = "https://export.arxiv.org/api/query"
USER_AGENT = "math-lw-literature-builder/0.1 (mailto:research@example.invalid)"
socket.setdefaulttimeout(45)

TARGET_TOTAL = 100
QUOTAS = {
    "kv_cache_compression": 45,
    "serving_systems": 15,
    "uncertainty_calibration": 20,
    "long_context_reliability": 12,
    "theory_interpretability": 8,
}

OPENALEX_QUERIES = {
    "kv_cache_compression": [
        '"KV cache" compression large language model',
        '"KV cache" eviction LLM inference',
        '"KV cache" quantization long context',
        '"key-value cache" transformer inference',
        '"attention sink" KV cache',
        '"heavy hitter" KV cache',
        '"LLM inference" cache compression',
    ],
    "serving_systems": [
        'large language model serving memory management paged attention',
        'LLM serving continuous batching KV cache',
        'efficient transformer inference serving cache',
        'long context LLM serving memory',
    ],
    "uncertainty_calibration": [
        'large language model uncertainty calibration',
        'selective prediction large language models uncertainty',
        'conformal prediction large language models risk control',
        'language model confidence calibration hallucination',
        'uncertainty estimation neural language generation',
    ],
    "long_context_reliability": [
        'long context large language model benchmark retrieval',
        'needle in a haystack long context language model',
        'long context LLM evaluation benchmark',
        'lost in the middle language models',
    ],
    "theory_interpretability": [
        'transformer attention interpretability token pruning',
        'attention heads long range dependencies transformer',
        'conformal risk control machine learning',
        'selective classification risk coverage',
    ],
}

ARXIV_QUERIES = {
    "kv_cache_compression": [
        'all:"KV cache" AND all:"large language model"',
        'all:"KV cache compression"',
        'all:"KV cache eviction"',
        'all:"KV cache quantization"',
        'all:"long context" AND all:"KV cache"',
        'all:"LLM inference" AND all:"cache"',
    ],
    "serving_systems": [
        'all:"LLM serving"',
        'all:"large language model serving"',
        'all:"PagedAttention"',
        'all:"continuous batching" AND all:"LLM"',
    ],
    "uncertainty_calibration": [
        'all:"large language model" AND all:"uncertainty"',
        'all:"language model" AND all:"calibration"',
        'all:"conformal prediction" AND all:"large language model"',
        'all:"conformal risk control"',
        'all:"selective prediction" AND all:"language model"',
    ],
    "long_context_reliability": [
        'all:"long context" AND all:"large language model" AND all:"benchmark"',
        'all:"needle in a haystack" AND all:"language model"',
        'all:"lost in the middle" AND all:"language model"',
        'all:"LongBench"',
        'all:"RULER" AND all:"language model"',
    ],
    "theory_interpretability": [
        'all:"transformer" AND all:"attention head" AND all:"interpretability"',
        'all:"attention is not explanation"',
        'all:"transformer circuits"',
        'all:"attention" AND all:"token pruning" AND all:"transformer"',
    ],
}

TITLE_SEEDS = {
    "kv_cache_compression": [
        "Efficient Memory Management for Large Language Model Serving with PagedAttention",
        "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models",
        "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time",
        "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving",
        "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization",
        "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
        "The Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs",
        "SnapKV: LLM Knows What You are Looking for Before Generation",
        "PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference",
        "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models",
        "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads",
        "RazorAttention: Efficient KV Cache Compression Through Retrieval Heads",
        "Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning",
        "NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time",
        "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification",
        "ThinK: Thinner Key Cache by Query-Driven Pruning",
        "Palu: Compressing KV-Cache with Low-Rank Projection",
        "ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression",
        "AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations",
        "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget",
        "KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches",
        "The Pitfalls of KV Cache Compression in Evaluating Long-Context Ability of LLMs",
        "Expected Attention: On the KV Cache Compression for Long Context LLM Inference",
        "The risk of KV cache compression",
        "CONF-KV: Uncertainty-Guided Compression for Conformal KV Cache in Long-Context LLMs",
    ],
    "serving_systems": [
        "Orca: A Distributed Serving System for Transformer-Based Generative Models",
        "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU",
        "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention",
        "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills",
        "Fast Distributed Inference Serving for Large Language Models",
        "SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference",
        "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale",
    ],
    "uncertainty_calibration": [
        "On Calibration of Modern Neural Networks",
        "Conformal Risk Control",
        "Language Models Mostly Know What They Know",
        "Calibrated Language Models Must Hallucinate",
        "Teaching Models to Express Their Uncertainty in Words",
        "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation",
        "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
        "Calibrated Confidence Estimation for Tabular Question Answering",
        "Know What You Don't Know: Unanswerable Questions for SQuAD",
        "Confidence Calibration for Question Answering",
    ],
    "long_context_reliability": [
        "Lost in the Middle: How Language Models Use Long Contexts",
        "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding",
        "NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?",
        "RULER: What's the Real Context Size of Your Long-Context Language Models?",
        "SCBench: A KV Cache-Centric Analysis of Long-Context Methods",
        "InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens",
        "Long Range Arena: A Benchmark for Efficient Transformers",
    ],
    "theory_interpretability": [
        "Attention Is All You Need",
        "Attention is not Explanation",
        "What Does BERT Look at? An Analysis of BERT's Attention",
        "Are Sixteen Heads Really Better than One?",
        "A Mathematical Perspective on Transformers",
        "A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations",
    ],
}

BLACKLIST_TITLES = {
    "Selective Prediction for Semantic Segmentation Using Conformal Prediction",
    "Trusted Uncertainty in Large Language Models: A Unified Framework for Confidence Calibration and Risk-Controlled Refusal",
    "Needle In A Haystack: Evaluating Long-Context LLMs",
    "Not All Needles Are Found: How Fact Distribution and Don't Make It Up Prompts Shape Literal Extraction, Logical Inference, and Long-Context Reliability",
    "A Mathematical Framework for Transformer Circuits",
    "Causal Scrubbing: a method for rigorously testing interpretability hypotheses",
}

MANUAL_PDF_URLS_RAW = {
    "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models": "https://arxiv.org/pdf/2306.14048",
    "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention": "https://arxiv.org/pdf/2309.06180",
    "SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference": "https://arxiv.org/pdf/2305.09781",
    "Orca: A Distributed Serving System for Transformer-Based Generative Models": "https://www.usenix.org/system/files/osdi22-yu.pdf",
    "The Pitfalls of KV Cache Compression in Evaluating Long-Context Ability of LLMs": "https://starai.cs.ucla.edu/papers/ChenArxiv25.pdf",
    "InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens": "https://arxiv.org/pdf/2402.13718",
    "Confidence Calibration for Question Answering": "https://assets.amazon.science/6d/70/c50b2eb141d3bcf1565e62b60211/qa-calibration-of-language-model-confidence-scores.pdf",
    "Calibrated Confidence Estimation for Tabular Question Answering": "https://arxiv.org/pdf/2604.12491",
    "CONF-KV: Uncertainty-Guided Compression for Conformal KV Cache in Long-Context LLMs": "https://arxiv.org/pdf/2605.24786",
    "Expected Attention: On the KV Cache Compression for Long Context LLM Inference": "https://arxiv.org/pdf/2510.00636",
    "A Mathematical Perspective on Transformers": "https://arxiv.org/pdf/2312.10794",
    "NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?": "https://arxiv.org/pdf/2407.11963",
}


@dataclass
class Paper:
    title: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    doi: str = ""
    openalex_id: str = ""
    url: str = ""
    pdf_url: str = ""
    arxiv_id: str = ""
    category: str = ""
    source: str = ""
    cited_by_count: int = 0
    relevance: int = 0
    abstract: str = ""
    pdf_path: str = ""
    pdf_status: str = "not_attempted"
    sha256: str = ""
    pages: int | None = None


def request_json(url: str, params: dict[str, str | int] | None = None, timeout: int = 15) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_text(url: str, params: dict[str, str | int] | None = None, timeout: int = 15) -> str:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def norm_title(title: str) -> str:
    title = unicodedata.normalize("NFKD", title or "")
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def slugify(title: str, limit: int = 70) -> str:
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:limit].strip("-") or "paper"


def abstract_from_inverted(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for idx in idxs:
            positions.append((idx, word))
    return " ".join(word for _, word in sorted(positions))


def author_names(authorships: list[dict[str, Any]]) -> list[str]:
    names = []
    for item in authorships or []:
        author = item.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
    return names


def venue_name(work: dict[str, Any]) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    if source.get("display_name"):
        return source["display_name"]
    return work.get("host_venue", {}).get("display_name") or ""


def pdf_from_locations(work: dict[str, Any]) -> str:
    for loc_key in ("best_oa_location", "primary_location"):
        loc = work.get(loc_key) or {}
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    return ""


def landing_url(work: dict[str, Any]) -> str:
    for loc_key in ("best_oa_location", "primary_location"):
        loc = work.get(loc_key) or {}
        if loc.get("landing_page_url"):
            return loc["landing_page_url"]
    return work.get("doi") or work.get("id") or ""


def from_openalex_work(work: dict[str, Any], category: str, source: str) -> Paper:
    return Paper(
        title=work.get("title") or "",
        year=work.get("publication_year"),
        authors=author_names(work.get("authorships") or []),
        venue=venue_name(work),
        doi=(work.get("doi") or "").replace("https://doi.org/", ""),
        openalex_id=work.get("id") or "",
        url=landing_url(work),
        pdf_url=pdf_from_locations(work),
        category=category,
        source=source,
        cited_by_count=int(work.get("cited_by_count") or 0),
        abstract=abstract_from_inverted(work.get("abstract_inverted_index")),
    )


def openalex_search(query: str, category: str, per_page: int = 50) -> list[Paper]:
    params = {
        "search": query,
        "per-page": per_page,
        "filter": "from_publication_date:2017-01-01",
        "select": "id,doi,title,publication_year,cited_by_count,authorships,primary_location,best_oa_location,abstract_inverted_index",
    }
    data = request_json(OPENALEX, params)
    return [from_openalex_work(w, category, f"openalex:{query}") for w in data.get("results", [])]


def openalex_title(title: str, category: str) -> Paper | None:
    clean_title = re.sub(r"[?:;]+", " ", title)
    params = {
        "search": clean_title,
        "per-page": 5,
        "select": "id,doi,title,publication_year,cited_by_count,authorships,primary_location,best_oa_location,abstract_inverted_index",
    }
    data = request_json(OPENALEX, params)
    wanted = norm_title(title)
    best: Paper | None = None
    best_score = -1
    for work in data.get("results", []):
        got = norm_title(work.get("title") or "")
        overlap = len(set(wanted.split()) & set(got.split()))
        score = overlap * 10 - abs(len(wanted) - len(got))
        if wanted == got:
            score += 1000
        if score > best_score:
            best_score = score
            best = from_openalex_work(work, category, f"openalex-title:{title}")
    if best_score < 25:
        return Paper(title=title, category=category, source="manual-title-seed")
    return best


def parse_arxiv_id(url_or_id: str) -> str:
    text = url_or_id or ""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?", text)
    if m:
        return m.group(1)
    m = re.search(r"\barXiv:([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?", text, re.I)
    if m:
        return m.group(1)
    return ""


def arxiv_title_lookup(title: str) -> tuple[str, str]:
    query = f'ti:"{title[:120]}"'
    try:
        xml = request_text(ARXIV, {"search_query": query, "start": 0, "max_results": 3})
    except Exception:
        return "", ""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    wanted = norm_title(title)
    best_id = ""
    best_title = ""
    best_score = -1
    for entry in root.findall("a:entry", ns):
        etitle = " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split())
        eid = entry.findtext("a:id", default="", namespaces=ns) or ""
        got = norm_title(etitle)
        score = len(set(wanted.split()) & set(got.split())) * 10 - abs(len(wanted) - len(got))
        if wanted == got:
            score += 1000
        if score > best_score:
            best_score = score
            best_id = parse_arxiv_id(eid)
            best_title = etitle
    if best_score >= 25 and best_id:
        return best_id, best_title
    return "", ""


def arxiv_entry_text(entry: ET.Element, name: str, ns: dict[str, str]) -> str:
    return " ".join((entry.findtext(f"a:{name}", default="", namespaces=ns) or "").split())


def arxiv_search(query: str, category: str, max_results: int = 40) -> list[Paper]:
    try:
        xml = request_text(
            ARXIV,
            {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=30,
        )
    except Exception as exc:
        print(f"[warn] arXiv search failed: {query}: {exc}", file=sys.stderr)
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("a:entry", ns):
        eid = entry.findtext("a:id", default="", namespaces=ns) or ""
        arxiv_id = parse_arxiv_id(eid)
        authors = [
            arxiv_entry_text(author, "name", ns)
            for author in entry.findall("a:author", ns)
            if arxiv_entry_text(author, "name", ns)
        ]
        published = arxiv_entry_text(entry, "published", ns)
        year = int(published[:4]) if published[:4].isdigit() else None
        title = arxiv_entry_text(entry, "title", ns)
        summary = arxiv_entry_text(entry, "summary", ns)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""
        papers.append(
            Paper(
                title=title,
                year=year,
                authors=authors,
                venue="arXiv",
                url=eid,
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                category=category,
                source=f"arxiv:{query}",
                abstract=summary,
            )
        )
    return papers


def score_paper(p: Paper) -> int:
    text = " ".join([p.title, p.abstract, p.venue]).lower()
    positive = [
        "kv cache", "key-value cache", "k v cache", "large language model", "llm",
        "transformer", "long context", "inference", "serving", "calibration",
        "uncertainty", "conformal", "selective", "risk", "attention",
        "quantization", "compression", "eviction", "pruning",
    ]
    negative = [
        "redis", "database", "wireless", "blockchain", "sensor", "internet of things",
        "vehicular", "image", "video compression", "cache hit rate",
        "fraud", "aml", "protein", "audio question answering", "object ownership",
        "social simulation", "edge devices", "vision-only", "fpga",
    ]
    score = 0
    for word in positive:
        if word in text:
            score += 8
    for word in negative:
        if word in text:
            score -= 25
    score += min(p.cited_by_count, 500) // 25
    if p.pdf_url or parse_arxiv_id(p.url):
        score += 20
    if "manual-title-seed" in p.source:
        score += 100
    if "scopus-top100" in p.source:
        score += 25
    if p.year and p.year >= 2023:
        score += 12
    return score


def merge_paper(existing: Paper, incoming: Paper) -> Paper:
    for attr in ("year", "venue", "doi", "openalex_id", "url", "pdf_url", "arxiv_id", "abstract"):
        if not getattr(existing, attr) and getattr(incoming, attr):
            setattr(existing, attr, getattr(incoming, attr))
    if not existing.authors and incoming.authors:
        existing.authors = incoming.authors
    existing.cited_by_count = max(existing.cited_by_count, incoming.cited_by_count)
    if len(incoming.source) < 250 and incoming.source not in existing.source:
        existing.source = f"{existing.source}; {incoming.source}" if existing.source else incoming.source
    return existing


def collect_candidates(use_openalex: bool = False) -> list[Paper]:
    papers: dict[str, Paper] = {}

    def add(p: Paper) -> None:
        if not p.title:
            return
        key = norm_title(p.title)
        if not key or len(key) < 8:
            return
        if key in {norm_title(title) for title in BLACKLIST_TITLES}:
            return
        p.relevance = score_paper(p)
        if key in papers:
            papers[key] = merge_paper(papers[key], p)
            papers[key].relevance = max(papers[key].relevance, p.relevance)
        else:
            papers[key] = p

    if SCOPUS_CANDIDATES.exists():
        for item in json.loads(SCOPUS_CANDIDATES.read_text()):
            title = item.get("title", "")
            source = f"scopus-top100-rank-{item.get('rank')}"
            add(Paper(title=title, year=int(item["year"]) if str(item.get("year", "")).isdigit() else None,
                      authors=[item.get("authors", "")] if item.get("authors") else [],
                      venue=item.get("source", ""), url=item.get("scopus_url", ""),
                      category="kv_cache_compression", source=source,
                      cited_by_count=int(str(item.get("citations", "0")).replace(",", "") or 0)))

    for category, titles in TITLE_SEEDS.items():
        for title in titles:
            print(f"[collect:title] {category}: {title[:80]}", flush=True)
            add(Paper(title=title, category=category, source="manual-title-seed"))
            if use_openalex:
                try:
                    paper = openalex_title(title, category)
                    if paper:
                        add(paper)
                except Exception as exc:
                    print(f"[warn] OpenAlex title lookup failed: {title}: {exc}", file=sys.stderr)
                time.sleep(0.35)

    for category, queries in ARXIV_QUERIES.items():
        for query in queries:
            print(f"[collect:arxiv] {category}: {query}", flush=True)
            for paper in arxiv_search(query, category, max_results=40):
                add(paper)
            time.sleep(3.1)

    if use_openalex:
        for category, queries in OPENALEX_QUERIES.items():
            for query in queries:
                print(f"[collect:openalex] {category}: {query}", flush=True)
                try:
                    for paper in openalex_search(query, category, per_page=25):
                        add(paper)
                except Exception as exc:
                    print(f"[warn] OpenAlex search failed: {query}: {exc}", file=sys.stderr)
                time.sleep(0.5)

    selected = []
    by_category = {cat: [] for cat in QUOTAS}
    for paper in papers.values():
        if paper.category not in by_category:
            paper.category = infer_category(paper)
        by_category.setdefault(paper.category, []).append(paper)
    for cat, quota in QUOTAS.items():
        ranked = sorted(by_category.get(cat, []), key=lambda p: (p.relevance, p.cited_by_count), reverse=True)
        selected.extend(ranked[: quota + 20])
    selected = dedupe(selected)
    selected = sorted(selected, key=lambda p: category_rank(p.category) * 100000 - p.relevance * 100 - p.cited_by_count)
    return selected


def infer_category(p: Paper) -> str:
    text = " ".join([p.title, p.abstract, p.venue]).lower()
    if any(k in text for k in ["uncertainty", "calibration", "conformal", "selective", "hallucinat"]):
        return "uncertainty_calibration"
    if any(k in text for k in ["long context", "needle", "lost in the middle", "longbench", "ruler"]):
        return "long_context_reliability"
    if any(k in text for k in ["serving", "pagedattention", "batching", "throughput"]):
        return "serving_systems"
    if any(k in text for k in ["interpret", "attention head", "transformer circuits"]):
        return "theory_interpretability"
    return "kv_cache_compression"


def category_rank(cat: str) -> int:
    return list(QUOTAS).index(cat) if cat in QUOTAS else 99


def dedupe(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    for p in papers:
        key = norm_title(p.title)
        if key in seen:
            seen[key] = merge_paper(seen[key], p)
        else:
            seen[key] = p
    return list(seen.values())


def filename_for(idx: int, p: Paper) -> str:
    year = str(p.year or "na")
    return f"{idx:03d}_{year}_{slugify(p.title)}.pdf"


def candidate_pdf_urls(p: Paper) -> list[str]:
    urls = []
    manual_urls = {norm_title(title): url for title, url in MANUAL_PDF_URLS_RAW.items()}
    manual_url = manual_urls.get(norm_title(p.title))
    if manual_url:
        urls.append(manual_url)
    for url in [p.pdf_url, p.url]:
        arxiv_id = parse_arxiv_id(url)
        if arxiv_id:
            p.arxiv_id = p.arxiv_id or arxiv_id
            urls.append(f"https://arxiv.org/pdf/{arxiv_id}")
        elif url and url.lower().endswith(".pdf"):
            urls.append(url)
    if not p.arxiv_id:
        arxiv_id, arxiv_title = arxiv_title_lookup(p.title)
        if arxiv_id:
            p.arxiv_id = arxiv_id
            if not p.title and arxiv_title:
                p.title = arxiv_title
    if p.arxiv_id:
        urls.append(f"https://arxiv.org/pdf/{p.arxiv_id}")
    if p.pdf_url:
        urls.append(p.pdf_url)
    unique = []
    for url in urls:
        if url and url not in unique:
            unique.append(url)
    return unique


def download_pdf(url: str, dest: Path) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        return False, f"http_{exc.code}"
    except TimeoutError:
        return False, "timeout"
    except socket.timeout:
        return False, "timeout"
    except Exception as exc:
        return False, f"error_{type(exc).__name__}"
    if len(data) < 10_000:
        return False, "too_small"
    if not data.lstrip().startswith(b"%PDF"):
        return False, "not_pdf"
    dest.write_bytes(data)
    return True, "ok"


def pdf_pages(path: Path) -> int | None:
    try:
        import subprocess

        out = subprocess.check_output(["pdfinfo", str(path)], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"^Pages:\s+(\d+)", out, re.M)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def bibtex_key(p: Paper, idx: int) -> str:
    author = "paper"
    if p.authors:
        first = p.authors[0]
        if isinstance(first, str):
            author = re.sub(r"[^A-Za-z]", "", first.split()[-1]) or "paper"
    title_word = next((w for w in re.findall(r"[A-Za-z]{4,}", p.title) if w.lower() not in {"with", "from", "that", "this", "large", "language", "model"}), "work")
    return f"{author}{p.year or 'na'}{title_word}{idx:03d}"


def write_outputs(papers: list[Paper]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (LIT / "catalog.json").write_text(json.dumps([p.__dict__ for p in papers], indent=2, ensure_ascii=False))
    with (LIT / "catalog.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["idx", "category", "year", "title", "authors", "venue", "doi", "url", "pdf_url", "pdf_path", "pages", "pdf_status", "cited_by_count", "relevance"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, p in enumerate(papers, 1):
            writer.writerow({
                "idx": idx,
                "category": p.category,
                "year": p.year or "",
                "title": p.title,
                "authors": "; ".join(p.authors[:8]),
                "venue": p.venue,
                "doi": p.doi,
                "url": p.url,
                "pdf_url": p.pdf_url,
                "pdf_path": p.pdf_path,
                "pages": p.pages or "",
                "pdf_status": p.pdf_status,
                "cited_by_count": p.cited_by_count,
                "relevance": p.relevance,
            })

    bib_entries = []
    for idx, p in enumerate(papers, 1):
        key = bibtex_key(p, idx)
        authors = " and ".join(p.authors) if p.authors else ""
        fields = {
            "title": p.title,
            "author": authors,
            "year": str(p.year or ""),
            "journal": p.venue,
            "doi": p.doi,
            "url": p.url or p.pdf_url,
        }
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
        bib_entries.append(f"@article{{{key},\n{body}\n}}\n")
    (LIT / "library.bib").write_text("\n".join(bib_entries), encoding="utf-8")

    by_cat: dict[str, int] = {}
    ok = 0
    for p in papers:
        by_cat[p.category] = by_cat.get(p.category, 0) + 1
        if p.pdf_status == "ok":
            ok += 1
    lines = [
        "# Literature Library",
        "",
        f"Total records: {len(papers)}",
        f"PDFs downloaded and structurally validated: {ok}",
        "",
        "## Categories",
        "",
    ]
    for cat in QUOTAS:
        lines.append(f"- {cat}: {by_cat.get(cat, 0)}")
    lines += [
        "",
        "## Files",
        "",
        "- `catalog.csv`: sortable reading catalog.",
        "- `catalog.json`: full metadata for scripts.",
        "- `library.bib`: BibTeX entries for the manuscript.",
        "- `pdfs/`: local PDF files, ignored by git.",
        "",
        "## Notes",
        "",
        "PDFs are downloaded only from open-access URLs discovered through public metadata services.",
        "Publisher/member access should be used manually for papers that are not open access.",
    ]
    (LIT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(target: int, dry_run: bool, use_openalex: bool) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(use_openalex=use_openalex)
    for p in candidates:
        p.category = p.category if p.category in QUOTAS else infer_category(p)
        p.relevance = score_paper(p)

    chosen: list[Paper] = []
    chosen_keys: set[str] = set()
    for cat, quota in QUOTAS.items():
        ranked = sorted(
            [p for p in candidates if p.category == cat],
            key=lambda p: (p.relevance, p.cited_by_count),
            reverse=True,
        )
        for p in ranked[:quota]:
            key = norm_title(p.title)
            if key not in chosen_keys:
                chosen.append(p)
                chosen_keys.add(key)

    if len(chosen) < target:
        ranked_all = sorted(candidates, key=lambda p: (p.relevance, p.cited_by_count), reverse=True)
        for p in ranked_all:
            key = norm_title(p.title)
            if key not in chosen_keys:
                chosen.append(p)
                chosen_keys.add(key)
            if len(chosen) >= target:
                break

    chosen = chosen[:target]

    for idx, p in enumerate(chosen, 1):
        p.relevance = score_paper(p)
        dest = PDF_DIR / filename_for(idx, p)
        if dry_run:
            p.pdf_status = "dry_run"
            continue
        if dest.exists() and dest.stat().st_size > 10_000:
            p.pdf_status = "ok"
            p.pdf_path = str(dest.relative_to(LIT))
            p.sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
            p.pages = pdf_pages(dest)
            continue
        statuses = []
        for url in candidate_pdf_urls(p):
            ok, status = download_pdf(url, dest)
            statuses.append(f"{status}:{url}")
            if ok:
                p.pdf_status = "ok"
                p.pdf_url = url
                p.pdf_path = str(dest.relative_to(LIT))
                p.sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
                p.pages = pdf_pages(dest)
                break
            time.sleep(0.2)
        if p.pdf_status != "ok":
            p.pdf_status = "failed|" + "|".join(statuses[:3])
        print(f"[{idx:03d}/{len(chosen)}] {p.pdf_status} {p.category}: {p.title[:90]}", flush=True)
        time.sleep(0.35)

    write_outputs(chosen)
    print(f"Wrote {len(chosen)} records to {LIT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-openalex", action="store_true")
    args = parser.parse_args()
    run(args.target, args.dry_run, args.use_openalex)


if __name__ == "__main__":
    main()
