from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

from src.chroma_store import fetch_all_records, update_metadata_values
from src.text_preprocessing import clean_text


GERMAN_STOP_WORDS = {
    "aber",
    "als",
    "am",
    "an",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bis",
    "da",
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "es",
    "für",
    "hat",
    "im",
    "in",
    "ist",
    "mit",
    "nach",
    "nicht",
    "oder",
    "sich",
    "und",
    "von",
    "war",
    "zu",
    "zum",
    "zur",
}
FIELD_STOP_WORDS = {
    "business",
    "cause",
    "claim",
    "country",
    "damage",
    "damaged",
    "description",
    "line",
    "loss",
    "object",
    "type",
    "anspruch",
    "beschreibung",
    "schaden",
    "schäden",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS).union(GERMAN_STOP_WORDS).union(FIELD_STOP_WORDS)
TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß-]{2,}")
EMBEDDING_FIELD_LABELS = [
    "Claim description",
    "Line of business",
    "Claim type",
    "Cause of loss",
    "Damaged object",
    "Country",
]
KEYWORD_FIELD_LABELS = ["Claim description", "Claim type", "Cause of loss", "Damaged object"]
FIELD_RE = re.compile(
    rf"(?P<label>{'|'.join(re.escape(label) for label in EMBEDDING_FIELD_LABELS)}):\s*"
    rf"(?P<value>.*?)(?=\s+(?:{'|'.join(re.escape(label) for label in EMBEDDING_FIELD_LABELS)}):|$)",
    re.IGNORECASE,
)


def load_cluster_input(collection: Any) -> tuple[list[str], list[str], list[dict[str, Any]], np.ndarray]:
    data = fetch_all_records(collection, include_embeddings=True)
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
    if embeddings.ndim != 2 or not len(ids):
        return ids, documents, metadatas, np.empty((0, 0), dtype=np.float32)
    return ids, documents, metadatas, embeddings


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) == 0:
        raise ValueError("No embeddings available for clustering.")
    actual_clusters = min(max(1, n_clusters), len(embeddings))
    normalized = normalize(embeddings)
    model = KMeans(n_clusters=actual_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(normalized)
    centers = normalize(model.cluster_centers_)
    return labels, centers


def assign_clusters(collection: Any, ids: list[str], labels: np.ndarray) -> None:
    updates = [{"cluster_id": int(label)} for label in labels]
    update_metadata_values(collection, ids, updates)


def build_cluster_summary(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> dict[str, Any]:
    normalized = normalize(embeddings)
    frame_rows = []
    for claim_id, document, metadata, label in zip(ids, documents, metadatas, labels):
        row = dict(metadata or {})
        row["id"] = claim_id
        row["document"] = document
        row["cluster_id"] = int(label)
        frame_rows.append(row)
    frame = pd.DataFrame(frame_rows)
    keyword_corpus = [keyword_text_from_document(document) for document in frame["document"].tolist()]
    clusters: list[dict[str, Any]] = []
    for cluster_id in sorted(frame["cluster_id"].unique()):
        cluster_frame = frame[frame["cluster_id"] == cluster_id]
        indices = cluster_frame.index.to_numpy()
        centroid = centers[int(cluster_id)]
        distances = 1.0 - normalized[indices] @ centroid
        representative_indices = indices[np.argsort(distances)[:3]]
        representatives = [
            {
                "claim_id": clean_text(frame.loc[idx].get("claim_id") or frame.loc[idx].get("id")),
                "description": claim_description_from_document(frame.loc[idx].get("document")),
            }
            for idx in representative_indices
        ]
        cluster_keyword_texts = [keyword_corpus[idx] for idx in indices]
        keywords = frequent_terms(cluster_keyword_texts, corpus_texts=keyword_corpus)
        label = " / ".join(keywords[:3]) if keywords else f"Cluster {cluster_id}"
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "label": label.title(),
                "size": int(len(cluster_frame)),
                "representative_claims": representatives,
                "frequent_terms": keywords,
                "common_metadata": common_metadata(cluster_frame),
            }
        )
    return {"clusters": clusters}


def build_cluster_map(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    coordinates, method, n_neighbors = project_embeddings_2d(embeddings)
    points = []
    for claim_id, document, metadata, label, coordinate in zip(ids, documents, metadatas, labels, coordinates):
        row = dict(metadata or {})
        points.append(
            {
                "claim_id": clean_text(row.get("claim_id") or claim_id),
                "cluster_id": int(label),
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "description": claim_description_from_document(document),
            }
        )
    return {
        "projection": {
            "method": method,
            "metric": "cosine",
            "random_state": 42,
            "n_neighbors": n_neighbors,
        },
        "points": points,
    }


def project_embeddings_2d(embeddings: np.ndarray) -> tuple[np.ndarray, str, int | None]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32), "deterministic_fallback", None
    if len(embeddings) < 3:
        return fallback_coordinates(len(embeddings)), "deterministic_fallback", None

    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("Install umap-learn to build cluster map artifacts.") from exc

    n_neighbors = min(15, max(2, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    coordinates = reducer.fit_transform(normalize(embeddings))
    return np.asarray(coordinates, dtype=np.float32), "UMAP", n_neighbors


def fallback_coordinates(count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    if count == 1:
        return np.asarray([[0.0, 0.0]], dtype=np.float32)
    return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)


def frequent_terms(texts: list[str], *, corpus_texts: list[str] | None = None, limit: int = 10) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_keyword_text(text))
    if not counter:
        return []

    corpus = corpus_texts or texts
    document_frequency: defaultdict[str, int] = defaultdict(int)
    for text in corpus:
        for token in set(tokenize_keyword_text(text)):
            document_frequency[token] += 1

    total_documents = max(1, len(corpus))
    scored = []
    for term, count in counter.items():
        idf = math.log((1 + total_documents) / (1 + document_frequency[term])) + 1
        scored.append((term, count * idf))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [term for term, _ in scored[:limit]]


def tokenize_keyword_text(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(clean_text(text).casefold())
        if token not in STOP_WORDS
    ]


def keyword_text_from_document(document: Any) -> str:
    fields = embedding_document_fields(document)
    if not fields:
        return clean_text(document)
    return " ".join(fields.get(label, "") for label in KEYWORD_FIELD_LABELS)


def claim_description_from_document(document: Any) -> str:
    fields = embedding_document_fields(document)
    return fields.get("Claim description", "") or clean_text(document)


def embedding_document_fields(document: Any) -> dict[str, str]:
    fields = {}
    for match in FIELD_RE.finditer(clean_text(document)):
        label = next(
            field_label
            for field_label in EMBEDDING_FIELD_LABELS
            if field_label.casefold() == match.group("label").casefold()
        )
        fields[label] = clean_text(match.group("value"))
    return fields


def common_metadata(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    fields = ["line_of_business", "claim_type", "cause_of_loss", "country", "claim_status", "loss_year"]
    result: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        if field not in frame:
            continue
        counts = frame[field].dropna().astype(str).value_counts().head(5)
        if not counts.empty:
            result[field] = [{"value": value, "count": int(count)} for value, count in counts.items()]
    return result


def write_cluster_artifact(path: Path, summary: dict[str, Any], *, n_clusters: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_clusters": n_clusters, **summary}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_cluster_map_artifact(path: Path, map_data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(map_data, indent=2, ensure_ascii=False), encoding="utf-8")
