# -*- coding: utf-8 -*-
"""
Gradio Interface for Text Deduplication & Search
Supports: FAISS, SimHash, MinHash
"""

import os
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from typing import Union, Set
import faiss
import re

# =============================================================================
# Device Configuration
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Hash Encoders
# =============================================================================

# class SimHashEncoder:
#     """
#     SimHash on embedding vectors using random hyperplanes.
#     """
#     def __init__(self, n_bits=128, normalize=True, orthogonal_planes=True, random_state=42):
#         self.n_bits = int(n_bits)
#         self.normalize = bool(normalize)
#         self.orthogonal_planes = bool(orthogonal_planes)
#         self.random_state = random_state
#         self.random_planes = None

#     def fit(self, X: np.ndarray):
#         X = np.asarray(X, dtype=np.float32)
#         n_features = X.shape[1]

#         rng = np.random.default_rng(self.random_state)
#         planes = rng.standard_normal(size=(self.n_bits, n_features)).astype(np.float32)

#         if self.orthogonal_planes:
#             Q, _ = np.linalg.qr(planes.T)
#             Q = Q[:, : self.n_bits]
#             self.random_planes = Q.T.astype(np.float32, copy=True)
#         else:
#             self.random_planes = planes

#         return self

#     def encode(self, X: np.ndarray) -> np.ndarray:
#         if self.random_planes is None:
#             raise RuntimeError("SimHashEncoder must be fitted before calling encode().")

#         X = np.asarray(X, dtype=np.float32)
#         if X.ndim == 1:
#             X = X[None, :]

#         if self.normalize:
#             norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
#             X = X / norms

#         projections = X @ self.random_planes.T
#         bits = (projections >= 0).astype(np.uint8)
#         return bits


# class MinHashEncoder:
#     """
#     MinHash on embeddings by converting vectors to sets of top-k dimension indices.
#     """
#     def __init__(self, n_hashes=128, topk_dims=20, random_state=42):
#         self.n_hashes = int(n_hashes)
#         self.topk_dims = int(topk_dims)
#         self.random_state = random_state
#         self.a = None
#         self.b = None
#         self.prime = None
#         self.n_features = None

#     def fit(self, X: np.ndarray):
#         X = np.asarray(X, dtype=np.float32)
#         self.n_features = X.shape[1]
#         self.prime = 2_000_003
#         rng = np.random.default_rng(self.random_state)
#         self.a = rng.integers(1, self.prime, size=self.n_hashes, dtype=np.int64)
#         self.b = rng.integers(0, self.prime, size=self.n_hashes, dtype=np.int64)
#         return self

#     def _vector_to_indices(self, x: np.ndarray) -> np.ndarray:
#         k = min(self.topk_dims, x.shape[0])
#         if k == x.shape[0]:
#             idx = np.arange(x.shape[0], dtype=np.int64)
#         else:
#             idx = np.argpartition(-np.abs(x), k - 1)[:k].astype(np.int64)
#         return np.sort(idx)

#     def encode(self, X: np.ndarray) -> np.ndarray:
#         X = np.asarray(X, dtype=np.float32)
#         if X.ndim == 1:
#             X = X[None, :]

#         n_samples = X.shape[0]
#         sigs = np.full(
#             (n_samples, self.n_hashes),
#             fill_value=np.iinfo(np.int32).max,
#             dtype=np.int32,
#         )

#         for i in range(n_samples):
#             idxs = self._vector_to_indices(X[i])
#             if idxs.size == 0:
#                 continue
#             hashes = (self.a[:, None] * idxs[None, :] + self.b[:, None]) % self.prime
#             mins = hashes.min(axis=1).astype(np.int32)
#             sigs[i] = mins

#         return sigs


# # =============================================================================
# # LSH Indexes
# # =============================================================================

# class SimHashLSHIndex:
#     """LSH for SimHash with banding + cosine re-rank."""
#     def __init__(
#         self,
#         n_bands: int = 8,
#         min_match_bands: int = 1,
#         use_cosine: bool = True,
#         max_rerank: int = 200,
#     ):
#         self.n_bands = int(n_bands)
#         self.min_match_bands = int(min_match_bands)
#         self.use_cosine = bool(use_cosine)
#         self.max_rerank = int(max_rerank)
#         self.buckets = None
#         self.docs_norm = None
#         self.signatures = None
#         self.band_width = None

#     def build(self, docs: np.ndarray, signatures: np.ndarray):
#         docs = np.asarray(docs, dtype=np.float32)
#         signatures = np.asarray(signatures, dtype=np.uint8)
#         self.signatures = signatures

#         norms = np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9
#         self.docs_norm = docs / norms

#         N, n_bits = signatures.shape
#         assert n_bits % self.n_bands == 0
#         self.band_width = n_bits // self.n_bands

#         reshaped_sigs = signatures.reshape(N, self.n_bands, self.band_width)
#         packed_sigs = np.packbits(reshaped_sigs, axis=2)

#         self.buckets = [defaultdict(list) for _ in range(self.n_bands)]
#         for b in range(self.n_bands):
#             keys_view = packed_sigs[:, b, :].reshape(N, -1)
#             for doc_id, key_arr in enumerate(keys_view):
#                 key_bytes = key_arr.tobytes()
#                 self.buckets[b][key_bytes].append(doc_id)

#     def _hamming_prefilter(self, q_bits: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
#         if self.max_rerank is None or len(candidate_ids) <= self.max_rerank:
#             return candidate_ids

#         q_bits = np.asarray(q_bits, dtype=np.uint8)
#         sig_sub = self.signatures[candidate_ids]
#         ham = np.count_nonzero(sig_sub != q_bits, axis=1)

#         top = min(self.max_rerank, len(ham))
#         top_idx = np.argpartition(ham, top - 1)[:top]
#         return candidate_ids[top_idx]

#     def query(self, query_vec: np.ndarray, top_k: int = 10, encoder=None):
#         if encoder is None:
#             raise ValueError("encoder must be provided.")
#         q_bits = encoder.encode(query_vec[None, :])[0]
#         return self.query_from_bits(q_bits, query_vec, top_k)

#     def query_from_bits(self, q_bits: np.ndarray, query_vec: np.ndarray = None, top_k: int = 10):
#         q_bits = np.asarray(q_bits, dtype=np.uint8).ravel()
#         n_bits = self.signatures.shape[1]
#         band_width = self.band_width

#         q_reshaped = q_bits.reshape(self.n_bands, band_width)
#         q_packed = np.packbits(q_reshaped, axis=1)

#         candidates_list = []
#         for b in range(self.n_bands):
#             key = q_packed[b].tobytes()
#             if key in self.buckets[b]:
#                 candidates_list.extend(self.buckets[b][key])

#         if not candidates_list:
#             return []

#         candidates_arr = np.array(candidates_list, dtype=int)
#         unique_ids, counts = np.unique(candidates_arr, return_counts=True)
#         mask = counts >= self.min_match_bands
#         final_candidates = unique_ids[mask]

#         if len(final_candidates) == 0:
#             final_candidates = unique_ids

#         final_candidates = self._hamming_prefilter(q_bits, final_candidates)

#         if (not self.use_cosine) or (query_vec is None):
#             sig_sub = self.signatures[final_candidates]
#             ham = np.count_nonzero(sig_sub != q_bits, axis=1)
#             sim = 1.0 - ham.astype(np.float32) / float(len(q_bits))

#             if len(sim) > top_k:
#                 top_idx = np.argpartition(-sim, top_k - 1)[:top_k]
#                 top_idx = top_idx[np.argsort(-sim[top_idx])]
#             else:
#                 top_idx = np.argsort(-sim)

#             return [(int(final_candidates[idx]), float(sim[idx])) for idx in top_idx]

#         cand_vectors = self.docs_norm[final_candidates]
#         q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
#         scores = cand_vectors @ q_norm

#         if len(scores) > top_k:
#             top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
#             top_idx = top_idx[np.argsort(-scores[top_idx])]
#         else:
#             top_idx = np.argsort(-scores)

#         return [(int(final_candidates[idx]), float(scores[idx])) for idx in top_idx]


# class MinHashLSHIndex:
#     """LSH for MinHash signatures with banding."""
#     def __init__(self, n_bands: int = 16, min_match_bands: int = 1):
#         self.n_bands = int(n_bands)
#         self.min_match_bands = int(min_match_bands)
#         self.buckets = None
#         self.signatures = None
#         self.band_width = None

#     def build(self, docs: np.ndarray, signatures: np.ndarray):
#         signatures = np.asarray(signatures, dtype=np.int32)
#         self.signatures = signatures
#         N, n_hashes = signatures.shape
#         assert n_hashes % self.n_bands == 0
#         self.band_width = n_hashes // self.n_bands

#         self.buckets = [defaultdict(list) for _ in range(self.n_bands)]
#         for doc_id in range(N):
#             sig = signatures[doc_id]
#             for b in range(self.n_bands):
#                 start = b * self.band_width
#                 end = start + self.band_width
#                 band = tuple(sig[start:end].tolist())
#                 self.buckets[b][band].append(doc_id)

#     def _collect_candidates(self, q_sig: np.ndarray) -> np.ndarray:
#         candidates = []
#         for b in range(self.n_bands):
#             start = b * self.band_width
#             end = start + self.band_width
#             key = tuple(q_sig[start:end].tolist())
#             if key in self.buckets[b]:
#                 candidates.extend(self.buckets[b][key])

#         if not candidates:
#             return np.array([], dtype=int)

#         cand_arr = np.array(candidates, dtype=int)
#         unique_ids, counts = np.unique(cand_arr, return_counts=True)
#         mask = counts >= self.min_match_bands
#         final = unique_ids[mask]
#         if final.size == 0:
#             final = unique_ids
#         return final

#     def query(self, query_vec: np.ndarray, top_k: int = 10, encoder=None):
#         if encoder is None:
#             raise ValueError("encoder must be provided.")
#         q_sig = encoder.encode(query_vec[None, :])[0]
#         return self.query_from_signature(q_sig, top_k)

#     def query_from_signature(self, q_sig: np.ndarray, top_k: int = 10):
#         q_sig = np.asarray(q_sig, dtype=np.int32).ravel()
#         final_candidates = self._collect_candidates(q_sig)
#         if final_candidates.size == 0:
#             return []

#         sig_sub = self.signatures[final_candidates]
#         sim = (sig_sub == q_sig[None, :]).mean(axis=1).astype(np.float32)

#         if len(sim) > top_k:
#             top_idx = np.argpartition(-sim, top_k - 1)[:top_k]
#             top_idx = top_idx[np.argsort(-sim[top_idx])]
#         else:
#             top_idx = np.argsort(-sim)

#         return [(int(final_candidates[idx]), float(sim[idx])) for idx in top_idx]


# # =============================================================================
# # Search Techniques
# # =============================================================================

# class FaissTechnique:
#     def __init__(self):
#         self.name = "faiss"
#         self.index = None
#         self.dim = None
#         self.docs_norm = None

#     def fit(self, docs: np.ndarray):
#         import faiss
#         docs = docs.astype("float32")
#         norms = np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9
#         self.docs_norm = docs / norms
#         self.dim = docs.shape[1]

#         self.index = faiss.IndexFlatIP(self.dim)
#         self.index.add(self.docs_norm)

#     def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
#         if self.index is None:
#             raise RuntimeError("FAISS index not fitted.")
#         q = query_vec.astype("float32")[None, :]
#         q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
#         distances, indices = self.index.search(q_norm, k)
#         return [(int(idx), float(d)) for d, idx in zip(distances[0], indices[0])]


# class SimHashTechnique:
#     def __init__(self, n_bits=256, n_bands=32, min_match_bands=5, max_rerank=500):
#         self.name = "simhash"
#         self.n_bits = n_bits
#         self.n_bands = n_bands
#         self.min_match_bands = min_match_bands
#         self.max_rerank = max_rerank
#         self.encoder = None
#         self.index = None

#     def fit(self, docs: np.ndarray):
#         docs = docs.astype("float32")
#         self.encoder = SimHashEncoder(n_bits=self.n_bits, normalize=True, orthogonal_planes=True)
#         self.encoder.fit(docs)
#         signatures = self.encoder.encode(docs)

#         self.index = SimHashLSHIndex(
#             n_bands=self.n_bands,
#             min_match_bands=self.min_match_bands,
#             use_cosine=True,
#             max_rerank=self.max_rerank,
#         )
#         self.index.build(docs, signatures)

#     def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
#         if self.index is None or self.encoder is None:
#             raise RuntimeError("SimHash index not fitted.")
#         return self.index.query(query_vec.astype("float32"), top_k=k, encoder=self.encoder)


# class MinHashTechnique:
#     def __init__(self, n_hashes=256, topk_dims=256, n_bands=64, min_match_bands=2):
#         self.name = "minhash"
#         self.n_hashes = n_hashes
#         self.topk_dims = topk_dims
#         self.n_bands = n_bands
#         self.min_match_bands = min_match_bands
#         self.encoder = None
#         self.index = None

#     def fit(self, docs: np.ndarray):
#         docs = docs.astype("float32")
#         self.encoder = MinHashEncoder(n_hashes=self.n_hashes, topk_dims=self.topk_dims)
#         self.encoder.fit(docs)
#         signatures = self.encoder.encode(docs)

#         self.index = MinHashLSHIndex(n_bands=self.n_bands, min_match_bands=self.min_match_bands)
#         self.index.build(docs, signatures)

#     def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
#         if self.index is None or self.encoder is None:
#             raise RuntimeError("MinHash index not fitted.")
#         return self.index.query(query_vec.astype("float32"), top_k=k, encoder=self.encoder)


class HashEncoder:
    def fit(self, docs: np.ndarray):
        raise NotImplementedError

    def encode(self, docs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# =============================
# 2. Index cho similarity search
# =============================

class BaseIndex:
    def build(self, docs: np.ndarray, signatures: np.ndarray = None):
        raise NotImplementedError

    def query(self, query_vec: np.ndarray, top_k: int = 10, encoder: HashEncoder = None):
        raise NotImplementedError


class ExactCosineIndex(BaseIndex):
    """Search exact theo cosine similarity (baseline)."""
    def build(self, docs: np.ndarray, signatures: np.ndarray = None):
        self.docs = docs
        norms = np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9
        self.normalized = docs / norms

    def query(self, query_vec: np.ndarray, top_k: int = 10, encoder: HashEncoder = None):
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = self.normalized @ q  # (N,)
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx]

# =============================================================================
# 3. Interface chung cho các kỹ thuật tìm kiếm
# =============================================================================

class BaseSearchTechnique:
    """
    Interface chung: nhận embedding vector, trả về (doc_id, score).
    """
    def __init__(self, name: str):
        self.name = name

    def fit(self, docs: np.ndarray) -> None:
        raise NotImplementedError

    def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError

"""### SimHash

"""

# =============================
# SimHash Encoder
# =============================

class SimHashEncoder:
    """
    SimHash encoder sử dụng random hyperplanes.
    Trả về packed bits để tiết kiệm RAM.
    """
    def __init__(self, n_bits=256, random_state=42):
        self.n_bits = n_bits
        self.random_state = random_state
        self.random_planes = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        # Tạo random hyperplanes (Gaussian)
        self.random_planes = rng.standard_normal(size=(self.n_bits, n_features), dtype=np.float32)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Trả về: Packed bits (n_samples, n_bytes). Tiết kiệm 8x RAM.
        """
        if self.random_planes is None:
            raise RuntimeError("Encoder chưa được fit.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]

        # Projections: (N, D) @ (D, n_bits) -> (N, n_bits)
        projections = X @ self.random_planes.T

        # Bits: 0 hoặc 1
        bits = (projections >= 0).astype(np.uint8)

        # Pack bits ngay lập tức: (N, n_bits) -> (N, n_bytes)
        return np.packbits(bits, axis=1)

# =============================
# SimHash LSH Index
# =============================

class SimHashLSHIndex:
    """
    LSH Index cho SimHash - Phiên bản KHÔNG có Cosine Reranking.
    Chỉ dùng Hamming distance để ranking.

    So với SimHashLSHIndex:
    - KHÔNG lưu docs_norm (tiết kiệm RAM)
    - KHÔNG tính cosine similarity
    - Ranking dựa 100% trên Hamming distance (thấp hơn = tốt hơn)
    """
    def __init__(self, n_bits, n_bands=32, min_match_bands=1, max_candidates=500):
        self.n_bits = n_bits
        self.n_bands = n_bands
        self.min_match_bands = min_match_bands
        self.max_candidates = max_candidates  # Giới hạn số candidates trả về

        # Kiểm tra tính hợp lệ
        self.n_bytes = n_bits // 8
        if self.n_bytes * 8 != n_bits:
            raise ValueError("n_bits phải chia hết cho 8.")

        if n_bands > self.n_bytes:
            raise ValueError(
                f"n_bands ({n_bands}) không thể lớn hơn n_bytes ({self.n_bytes}). "
                f"Với n_bits={n_bits}, n_bands tối đa là {self.n_bytes}."
            )

        self.bytes_per_band = self.n_bytes // n_bands

        if self.bytes_per_band == 0:
            raise ValueError(
                f"bytes_per_band = 0. n_bits={n_bits} quá nhỏ cho n_bands={n_bands}. "
                f"Cần n_bits >= {n_bands * 8}."
            )

        # Cấu trúc dữ liệu dạng mảng phẳng (Flat Arrays)
        self.bands_data = []
        self.sig_packed = None

        # Popcount table cho Hamming distance
        self._popcnt = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    def build(self, docs: np.ndarray, packed_signatures: np.ndarray):
        """
        Dùng sort để gom nhóm bucket.
        NOTE: docs parameter giữ lại để tương thích API, nhưng KHÔNG dùng.
        """
        self.sig_packed = packed_signatures

        # Xây dựng Index theo từng Band bằng Sort
        self.bands_data = []
        dtype_void = np.dtype((np.void, self.bytes_per_band))

        for b in range(self.n_bands):
            start = b * self.bytes_per_band
            end = (b + 1) * self.bytes_per_band

            keys = self.sig_packed[:, start:end]
            keys_void = np.ascontiguousarray(keys).view(dtype_void).ravel()
            sort_idx = np.argsort(keys_void)

            sorted_keys = keys_void[sort_idx]
            changes = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1], [True]))
            start_indices = np.where(changes)[0]
            unique_keys = sorted_keys[start_indices[:-1]]

            self.bands_data.append({
                'keys': unique_keys,
                'starts': start_indices,
                'doc_ids': sort_idx
            })

        print(f"[SimHashLSHIndex] Built {self.n_bands} bands.")

    def query(self, query_vec, encoder, top_k=10):
        """
        Query chỉ dùng Hamming distance để ranking.
        Score = 1 - (hamming_dist / n_bits) để score cao hơn = tốt hơn
        """
        # 1. Encode
        q_packed = encoder.encode(query_vec)[0]  # (n_bytes,)

        candidates_list = []
        dtype_void = np.dtype((np.void, self.bytes_per_band))

        # 2. Collect Candidates dùng Binary Search
        for b in range(self.n_bands):
            start_byte = b * self.bytes_per_band
            end_byte = (b + 1) * self.bytes_per_band

            q_key_bytes = q_packed[start_byte:end_byte]
            q_key_void = np.array([q_key_bytes.tobytes()], dtype=dtype_void)[0]

            band_struct = self.bands_data[b]
            keys = band_struct['keys']

            idx = np.searchsorted(keys, q_key_void)

            if idx < len(keys) and keys[idx] == q_key_void:
                start_pos = band_struct['starts'][idx]
                end_pos = band_struct['starts'][idx+1] if (idx + 1) < len(keys) else len(band_struct['doc_ids'])
                chunk_ids = band_struct['doc_ids'][start_pos:end_pos]
                candidates_list.append(chunk_ids)

        if not candidates_list:
            return []

        # 3. Merge & Voting
        all_candidates = np.concatenate(candidates_list)

        if self.min_match_bands > 1:
            if all_candidates.size == 0:
                return []
            counts = np.bincount(all_candidates)
            candidate_ids = np.where(counts >= self.min_match_bands)[0]
        else:
            candidate_ids = np.unique(all_candidates)

        if len(candidate_ids) == 0:
            return []

        # 4. Hamming Distance Ranking (ONLY - NO COSINE)
        candidate_sigs = self.sig_packed[candidate_ids]
        xor_result = np.bitwise_xor(candidate_sigs, q_packed)
        hamming_dists = self._popcnt[xor_result].sum(axis=1)

        # Convert Hamming distance to similarity score: score = 1 - (dist / n_bits)
        # Score càng cao càng giống (Hamming distance càng thấp càng tốt)
        hamming_scores = 1.0 - (hamming_dists.astype(np.float32) / self.n_bits)

        # 5. Top-K by Hamming Score (descending)
        if len(hamming_scores) > top_k:
            top_idx = np.argpartition(-hamming_scores, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-hamming_scores[top_idx])]
        else:
            top_idx = np.argsort(-hamming_scores)

        return [(int(candidate_ids[i]), float(hamming_scores[i])) for i in top_idx]

# # =============================
# # Complete Search Techniques
# # =============================
class SimHashTechnique(BaseSearchTechnique):
    """
    SimHash + LSH trên embedding - KHÔNG CÓ Cosine Reranking.
    Chỉ dùng Hamming distance để ranking.

    Ưu điểm:
    - Nhanh hơn (không cần tính cosine)
    - Tiết kiệm RAM (không lưu normalized docs)
    - Phù hợp khi chỉ cần approximate similarity

    Nhược điểm:
    - Độ chính xác thấp hơn SimHashTechnique (có Cosine rerank)
    """
    def __init__(
        self,
        n_bits: int = 256,
        n_bands: int = 32,
        min_match_bands: int = 5,
        max_candidates: int = 500,
    ):
        super().__init__(name="simhash")
        self.n_bits = n_bits
        self.n_bands = n_bands
        self.min_match_bands = min_match_bands
        self.max_candidates = max_candidates
        self.encoder = None
        self.index = None

    def fit(self, docs: np.ndarray) -> None:
        # 1) Fit encoder + encode docs
        encoder = SimHashEncoder(
            n_bits=self.n_bits,
            random_state=42,
        )
        encoder.fit(docs)
        signatures = encoder.encode(docs)

        # 2) Build LSH index (NO RERANK version)
        index = SimHashLSHIndex(
            n_bits=self.n_bits,
            n_bands=self.n_bands,
            min_match_bands=self.min_match_bands,
            max_candidates=self.max_candidates,
        )
        index.build(docs, signatures)

        self.encoder = encoder
        self.index = index
        print(f"[SimHash] Index built: n_bits={self.n_bits}, n_bands={self.n_bands}")

    def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.index is None or self.encoder is None:
            raise RuntimeError("SimHash index chưa fit.")
        neighbors = self.index.query(
            query_vec.astype("float32"),
            top_k=k,
            encoder=self.encoder,
        )
        return neighbors

"""### MinHash Embeddings"""

# =============================
# MinHash Embed Encoder
# =============================

class MinHashEmbedEncoder:
    """
    MinHash trên embedding:
    - Chuyển mỗi vector thành một "tập" các chiều top-k có |value| lớn nhất.
    - Áp dụng n_hashes hàm băm tuyến tính trên chỉ số chiều để tạo chữ ký MinHash.
    """
    def __init__(self, n_hashes=128, topk_dims=20, random_state=42):
        self.n_hashes = int(n_hashes)
        self.topk_dims = int(topk_dims)
        self.random_state = random_state
        self.a = None
        self.b = None
        self.prime = None
        self.n_features = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        self.n_features = X.shape[1]
        # Chọn một số nguyên tố lớn
        self.prime = 2_000_003
        rng = np.random.default_rng(self.random_state)
        self.a = rng.integers(1, self.prime, size=self.n_hashes, dtype=np.int64)
        self.b = rng.integers(0, self.prime, size=self.n_hashes, dtype=np.int64)
        return self

    def _vector_to_indices(self, x: np.ndarray) -> np.ndarray:
        """Lấy top-k chỉ số chiều theo |value|."""
        k = min(self.topk_dims, x.shape[0])
        if k == x.shape[0]:
            idx = np.arange(x.shape[0], dtype=np.int64)
        else:
            idx = np.argpartition(-np.abs(x), k - 1)[:k].astype(np.int64)
        return np.sort(idx)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, n_features)
        Trả về: signatures MinHash shape (n_samples, n_hashes) kiểu int32.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]

        n_samples = X.shape[0]
        sigs = np.full(
            (n_samples, self.n_hashes),
            fill_value=np.iinfo(np.int32).max,
            dtype=np.int32,
        )

        for i in range(n_samples):
            idxs = self._vector_to_indices(X[i])
            if idxs.size == 0:
                continue
            # hashes: (n_hashes, len(idxs))
            hashes = (self.a[:, None] * idxs[None, :] + self.b[:, None]) % self.prime
            mins = hashes.min(axis=1).astype(np.int32)
            sigs[i] = mins

        return sigs

# =============================
# MinHash Embed LSH Index
# =============================

class MinHashEmbedLSHIndex(BaseIndex):
    """LSH cho MinHash signatures với banding (buckets theo từng band)."""
    def __init__(self, n_bands: int = 16, min_match_bands: int = 1):
        self.n_bands = int(n_bands)
        self.min_match_bands = int(min_match_bands)
        self.bands_data = []
        self.signatures = None
        self.band_width = None

    def build(self, docs: np.ndarray, signatures: np.ndarray):
        """
        docs: (N, D) - không dùng trực tiếp, chỉ để giữ API cho thống nhất.
        signatures: (N, n_hashes) - MinHash signatures kiểu int32.
        """
        signatures = np.asarray(signatures, dtype=np.int32)
        self.signatures = signatures
        N, n_hashes = signatures.shape
        assert n_hashes % self.n_bands == 0, "n_hashes phải chia hết cho n_bands"
        self.band_width = n_hashes // self.n_bands

        # Build sort-based index per band
        self.bands_data = []
        dtype_void = np.dtype((np.void, self.band_width * 4))  # 4 bytes per int32

        for b in range(self.n_bands):
            start = b * self.band_width
            end = start + self.band_width

            # Extract band keys for all docs
            keys = self.signatures[:, start:end]
            keys_void = np.ascontiguousarray(keys).view(dtype_void).ravel()

            # Sort by keys
            sort_idx = np.argsort(keys_void, kind='stable')
            sorted_keys = keys_void[sort_idx]

            # Find bucket boundaries
            changes = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1], [True]))
            start_indices = np.where(changes)[0]
            unique_keys = sorted_keys[start_indices[:-1]]

            # Store compact structure
            self.bands_data.append({
                'keys': unique_keys,
                'starts': start_indices,
                'doc_ids': sort_idx
            })

        print(f"[MinHashEmbed-Optimized] Built {self.n_bands} bands with sort-based storage (Memory optimized)")

    def _collect_candidates(self, q_sig: np.ndarray) -> np.ndarray:
        """Collect candidates using binary search on sorted arrays."""
        candidates_list = []
        dtype_void = np.dtype((np.void, self.band_width * 4))

        for b in range(self.n_bands):
            start = b * self.band_width
            end = start + self.band_width

            q_key_bytes = q_sig[start:end]
            q_key_void = np.array([q_key_bytes.tobytes()], dtype=dtype_void)[0]

            band_struct = self.bands_data[b]
            keys = band_struct['keys']

            # Binary search
            idx = np.searchsorted(keys, q_key_void)

            if idx < len(keys) and keys[idx] == q_key_void:
                start_pos = band_struct['starts'][idx]
                end_pos = band_struct['starts'][idx + 1] if (idx + 1) < len(keys) else len(band_struct['doc_ids'])
                chunk_ids = band_struct['doc_ids'][start_pos:end_pos]
                candidates_list.append(chunk_ids)

        if not candidates_list:
            return np.array([], dtype=int)

        # Merge & voting
        cand_arr = np.concatenate(candidates_list)

        if self.min_match_bands > 1:
            unique_ids, counts = np.unique(cand_arr, return_counts=True)
            mask = counts >= self.min_match_bands
            final = unique_ids[mask]
            if final.size == 0:
                final = unique_ids
        else:
            final = np.unique(cand_arr)

        return final

    def query_from_signature(self, q_sig: np.ndarray, top_k: int = 10):
        """Query trực tiếp từ chữ ký MinHash q_sig."""
        q_sig = np.asarray(q_sig, dtype=np.int32).ravel()
        if self.signatures is None:
            raise RuntimeError("Index chưa được build với signatures.")
        assert q_sig.shape[0] == self.signatures.shape[1]

        final_candidates = self._collect_candidates(q_sig)
        if final_candidates.size == 0:
            return []

        sig_sub = self.signatures[final_candidates]
        # Ước lượng Jaccard ~ tỉ lệ số dòng trùng nhau trong chữ ký
        sim = (sig_sub == q_sig[None, :]).mean(axis=1).astype(np.float32)

        if len(sim) > top_k:
            top_idx = np.argpartition(-sim, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-sim[top_idx])]
        else:
            top_idx = np.argsort(-sim)

        results = []
        for idx in top_idx:
            doc_id = int(final_candidates[idx])
            score = float(sim[idx])
            results.append((doc_id, score))
        return results

    def query(self, query_vec: np.ndarray, top_k: int = 10, encoder: HashEncoder = None):
        if encoder is None:
            raise ValueError("encoder must be provided to encode query_vec to MinHash signature.")
        q_sig = encoder.encode(query_vec[None, :])[0]
        return self.query_from_signature(q_sig, top_k=top_k)

class MinHashEmbedTechnique(BaseSearchTechnique):
    """
    MinHash + LSH trên embedding.
    """
    def __init__(
        self,
        n_hashes: int = 256,
        topk_dims: int = 256,
        n_bands: int = 64,
        min_match_bands: int = 2,
    ):
        super().__init__(name="minhash_embed")
        self.n_hashes = n_hashes
        self.topk_dims = topk_dims
        self.n_bands = n_bands
        self.min_match_bands = min_match_bands
        self.encoder = None
        self.index = None

    def fit(self, docs: np.ndarray) -> None:
        # 1) Fit encoder
        encoder = MinHashEmbedEncoder(
            n_hashes=self.n_hashes,
            topk_dims=self.topk_dims,
            random_state=42,
        )
        encoder.fit(docs)
        signatures = encoder.encode(docs)

        # 2) Build LSH index
        index = MinHashEmbedLSHIndex(
            n_bands=self.n_bands,
            min_match_bands=self.min_match_bands,
        )
        index.build(docs, signatures)

        self.encoder = encoder
        self.index = index
        print(f"[MinHash Embed] Index built: n_hashes={self.n_hashes}, n_bands={self.n_bands}")

    def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.index is None or self.encoder is None:
            raise RuntimeError("MinHash Embed index chưa fit.")
        return self.index.query(query_vec, top_k=k, encoder=self.encoder)

"""### Faiss

"""

# =============================
# FAISS Techniques
# =============================

class FaissExactTechnique(BaseSearchTechnique):
    """
    FAISS Exact Search (Ground Truth).
    IndexFlatIP = Brute-force Inner Product.
    """
    def __init__(self):
        super().__init__(name="faiss_exact")
        self.index = None

    def fit(self, docs: np.ndarray) -> None:
        docs = docs.astype("float32")
        # Normalize để Inner Product trở thành Cosine Similarity
        faiss.normalize_L2(docs)
        self.dim = docs.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(docs)
        print(f"[FAISS Exact] Index built. Total: {self.index.ntotal}")

    def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        q = query_vec.astype("float32")[None, :]
        faiss.normalize_L2(q)

        distances, indices = self.index.search(q, k)

        neighbors = []
        for d, idx in zip(distances[0], indices[0]):
            neighbors.append((int(idx), float(d)))
        return neighbors


class FaissLSHTechnique(BaseSearchTechnique):
    """
    FAISS LSH Index.
    Đây là implementation SimHash của FAISS.
    """
    def __init__(self, n_bits=128):
        super().__init__(name="faiss_lsh")
        self.index = None
        self.n_bits = n_bits

    def fit(self, docs: np.ndarray) -> None:
        docs = docs.astype("float32")
        faiss.normalize_L2(docs)
        self.dim = docs.shape[1]

        # Implementation SimHash của FAISS
        self.index = faiss.IndexLSH(self.dim, self.n_bits)
        self.index.train(docs)
        self.index.add(docs)
        print(f"[FAISS LSH] Index built. Bits: {self.n_bits}")

    def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        q = query_vec.astype("float32")[None, :]
        faiss.normalize_L2(q)

        distances, indices = self.index.search(q, k)

        # Chuyển distance sang similarity score:
        # FAISS LSH trả về Hamming distance (số bits khác nhau)
        # Similarity = 1 - (hamming_distance / n_bits)
        neighbors = []
        for d, idx in zip(distances[0], indices[0]):
            similarity = 1.0 - (float(d) / self.n_bits)
            neighbors.append((int(idx), similarity))
        return neighbors

"""### Minhash Shingle"""

# =============================
# MinHash Shingle Encoder
# =============================

class MinHashShingleEncoder:
    """
    MinHash encoder cho short sentences sử dụng 64-bit stable shingle hashing.
    Dựa trên word shingles thay vì embedding vectors.
    """
    def __init__(
        self,
        num_perm: int = 256,
        ngram_size: int = 1,
        seed: int = 42,
        use_stopwords: bool = True,
        stopwords: Optional[Set[str]] = None,
    ):
        self.num_perm = int(num_perm)
        self.ngram_size = int(ngram_size)
        self.seed = int(seed)
        self.use_stopwords = use_stopwords

        # Cấu hình Stopwords
        if stopwords is None and use_stopwords:
            self.stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'must', 'can', 'it', 'this',
                'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'
            }
        else:
            self.stopwords = stopwords if stopwords is not None else set()

        # Khởi tạo các hệ số hoán vị (Permutation Coefficients)
        # Sử dụng Mersenne Prime lớn để hash phân bố đều
        self.mersenne_prime = (1 << 61) - 1
        rng = np.random.default_rng(self.seed)
        self.a = rng.integers(1, self.mersenne_prime, size=(1, self.num_perm), dtype=np.uint64)
        self.b = rng.integers(0, self.mersenne_prime, size=(1, self.num_perm), dtype=np.uint64)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    @staticmethod
    def _hash64(s: str) -> int:
        # Hash string thành số nguyên 64-bit
        # Lưu ý: hash() của Python không stable qua các lần restart session.
        # Nếu cần lưu index xuống đĩa dùng lâu dài, hãy thay bằng murmurhash hoặc xxhash.
        return hash(s) & 0xFFFFFFFFFFFFFFFF

    def _get_shingles(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        n = len(tokens)
        if n == 0:
            return np.array([], dtype=np.uint64)

        k = min(self.ngram_size, n)
        shingles = []
        for i in range(n - k + 1):
            ngram = " ".join(tokens[i : i + k])
            shingles.append(self._hash64(ngram))

        return np.unique(np.array(shingles, dtype=np.uint64))

    def encode(self, docs: Union[str, List[str], np.ndarray]) -> np.ndarray:
        if isinstance(docs, str):
            docs_list = [docs]
        else:
            docs_list = list(docs)

        N = len(docs_list)
        # Tính toán trên không gian 64-bit
        sigs = np.full((N, self.num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)

        p = self.mersenne_prime
        a = self.a
        b = self.b

        for i, doc in enumerate(docs_list):
            sh = self._get_shingles(doc) # (S,)
            if sh.size == 0:
                continue

            # Vector hóa phép tính Hash: (a*x + b) % p
            # sh_col: (S, 1)
            sh_col = (sh % p).reshape(-1, 1)

            # Broadcast: (S, 1) * (1, P) -> (S, P)
            # Min over axis 0 -> (P,)
            hash_vals = (sh_col * a + b) % p
            sigs[i] = hash_vals.min(axis=0)

        return sigs.astype(np.uint32)

# =============================
# MinHash Shingle Index
# =============================

class MinHashShingleIndex:
    """
    Sử dụng Sort-based LSH + Vectorized Hashing.
    - Build time cực nhanh (< 0.1s cho 100k docs).
    - Query time cực nhanh (Binary Search).
    """
    def __init__(
        self,
        num_perm: int = 256,
        threshold: float = 0.45,
        b: Optional[int] = None,
        r: Optional[int] = None,
        min_band_matches: int = 1,
        seed: int = 42
    ):
        self.num_perm = int(num_perm)
        self.threshold = float(threshold)
        self.min_band_matches = int(min_band_matches)
        self.seed = seed

        # Tự động chọn b và r tối ưu nếu không cung cấp
        if b is None or r is None:
            self.b, self.r = self._choose_br(self.num_perm, self.threshold)
        else:
            self.b, self.r = int(b), int(r)

        self.used_perm = self.b * self.r
        if self.used_perm > self.num_perm:
             raise ValueError(f"b*r ({self.used_perm}) must be <= num_perm ({self.num_perm})")

        print(f"[LSH Config] Bands={self.b}, Rows={self.r} (Used {self.used_perm}/{self.num_perm} perms)")

        # Dữ liệu index
        self.bands_data = []
        self.signatures: Optional[np.ndarray] = None
        self.hash_coeffs: Optional[np.ndarray] = None

    @staticmethod
    def _choose_br(num_perm: int, threshold: float) -> Tuple[int, int]:
        best = None # (error, -used, b, r)
        for r in range(1, num_perm + 1):
            b = num_perm // r
            if b == 0: continue
            t_est = (1.0 / b) ** (1.0 / r)
            error = abs(t_est - threshold)
            # Ưu tiên sai số thấp, sau đó ưu tiên dùng nhiều perm nhất
            key = (error, - (b*r), b, r)
            if best is None or key < best:
                best = key
        return best[2], best[3]

    def build(self, signatures: np.ndarray) -> None:
        """
        Xây dựng index siêu tốc bằng Vectorized Hashing (Dot Product).
        """
        # 1. Lưu trữ Compact
        self.signatures = np.ascontiguousarray(signatures[:, :self.used_perm], dtype=np.uint32)
        N = self.signatures.shape[0]
        self.bands_data = []

        # 2. Tạo ma trận ngẫu nhiên để Hash các bands (Vectorized Trick)
        rng = np.random.default_rng(self.seed)
        # hash_coeffs: (bands, rows) - Kiểu uint64 để tránh tràn số quá sớm khi nhân
        self.hash_coeffs = rng.integers(0, np.iinfo(np.uint64).max, size=(self.b, self.r), dtype=np.uint64)

        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r

            # Lấy data của band (N, r), ép kiểu lên uint64 để tính toán
            band_matrix = self.signatures[:, start:end].astype(np.uint64)
            coeffs = self.hash_coeffs[band_idx] # (r,)

            # --- KEY OPTIMIZATION: Matrix Multiplication thay vì Loop ---
            # (N, r) dot (r,) -> (N,)
            keys = np.dot(band_matrix, coeffs)

            # Sort-based Indexing
            sort_idx = np.argsort(keys, kind='stable') # uint64
            sorted_keys = keys[sort_idx]

            # Tìm ranh giới các buckets
            changes = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1], [True]))
            start_indices = np.where(changes)[0]
            unique_keys = sorted_keys[start_indices[:-1]]

            # Lưu trữ tối ưu bộ nhớ (uint32 cho indices)
            self.bands_data.append({
                'keys': unique_keys,                       # uint64 (Hash Values)
                'starts': start_indices.astype(np.uint32), # uint32 (Pointer)
                'doc_ids': sort_idx.astype(np.uint32)      # uint32 (Doc IDs)
            })

    def query(self, query_sig: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.signatures is None:
            return []

        # Query signature cũng cần ở dạng uint32
        q = np.asarray(query_sig, dtype=np.uint32)[:self.used_perm]

        candidates_list = []

        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r

            # Hash query band bằng coeffs đã lưu
            band_vec = q[start:end].astype(np.uint64)
            coeffs = self.hash_coeffs[band_idx]
            q_key = np.dot(band_vec, coeffs) # Scalar uint64

            # Binary Search
            band_struct = self.bands_data[band_idx]
            keys = band_struct['keys']

            idx = np.searchsorted(keys, q_key)

            if idx < len(keys) and keys[idx] == q_key:
                # Lấy danh sách ID trong bucket này
                start_pos = band_struct['starts'][idx]
                end_pos = band_struct['starts'][idx + 1] if (idx + 1) < len(keys) else len(band_struct['doc_ids'])
                chunk_ids = band_struct['doc_ids'][start_pos:end_pos]
                candidates_list.append(chunk_ids)

        if not candidates_list:
            return []

        # Gom ứng viên
        all_candidates = np.concatenate(candidates_list)

        # Lọc theo min_band_matches
        if self.min_band_matches > 1:
            if all_candidates.size == 0: return []
            counts = np.bincount(all_candidates, minlength=self.signatures.shape[0])
            candidate_ids = np.where(counts >= self.min_band_matches)[0]
        else:
            candidate_ids = np.unique(all_candidates)

        if len(candidate_ids) == 0:
            return []

        # --- SCORING STEP: Tính Jaccard ước lượng bằng full signatures ---
        cand_sigs = self.signatures[candidate_ids] # (C, used_perm)
        matches = (cand_sigs == q)
        scores = matches.mean(axis=1).astype(np.float32)

        # Top-K filtering
        if len(scores) > top_k:
            # Dùng argpartition để lấy top k nhanh nhất (không cần sort toàn bộ)
            top_part = np.argpartition(-scores, top_k)[:top_k]
            # Sort lại top k phần tử này
            final_order = top_part[np.argsort(-scores[top_part])]
        else:
            final_order = np.argsort(-scores)

        results = []
        for i in final_order:
            if scores[i] > 0.0:
                results.append((int(candidate_ids[i]), float(scores[i])))

        return results

# ==========================================
# 3. Wrapper Class (Giao diện chính)
# ==========================================
class MinHashShingleTechnique:
    def __init__(
        self,
        name: str = "minhash_shingle",
        # Encoder Params
        num_perm: int = 256,
        ngram_size: int = 1,
        seed: int = 42,
        use_stopwords: bool = False,
        stopwords: Optional[Set[str]] = None,
        # Index Params
        threshold: float = 0.45,
        min_band_matches: int = 1,
        b: Optional[int] = None,
        r: Optional[int] = None,
    ):
        self.name = name

        self.encoder = MinHashShingleEncoder(
            num_perm=num_perm,
            ngram_size=ngram_size,
            seed=seed,
            use_stopwords=use_stopwords,
            stopwords=stopwords
        )

        self.index = MinHashShingleIndex(
            num_perm=num_perm,
            threshold=threshold,
            min_band_matches=min_band_matches,
            b=b,
            r=r,
            seed=seed
        )

        self.docs = None
        self.preprocessing_time = 0.0
        self.indexing_time = 0.0

    def fit(self, docs: np.ndarray) -> None:
        print(f"[{self.name}] Starting process for {len(docs)} docs...")
        self.docs = docs

        # 1. Encoding (Tốn thời gian nhất - CPU Bound)
        t0 = time.perf_counter()
        signatures = self.encoder.encode(docs)
        self.preprocessing_time = time.perf_counter() - t0
        print(f"[{self.name}] Encoding done: {self.preprocessing_time:.4f}s")

        # 2. Indexing (Siêu nhanh - Vectorized)
        t1 = time.perf_counter()
        self.index.build(signatures)
        self.indexing_time = time.perf_counter() - t1
        print(f"[{self.name}] Indexing done: {self.indexing_time:.4f}s")

        # Cleanup: Xóa biến signatures tạm để giải phóng RAM
        del signatures

    def top_k(self, query: Union[str, np.ndarray], k: int = 10) -> List[Tuple[int, float]]:
        # Xử lý query
        if isinstance(query, str):
            # Encode trả về (1, num_perm)
            query_sig = self.encoder.encode([query])[0]
        else:
            query_sig = query

        return self.index.query(query_sig, top_k=k)

# =============================================================================
# Text Processing & Reformatting
# =============================================================================

def split_string_by_sentences(text: str, tokenizer, max_length: int) -> List[str]:
    """Split long text into smaller chunks based on sentence boundaries."""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
    except:
        # Fallback: simple split by period
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        if not sentences:
            sentences = [text]
    else:
        sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        tokenized = tokenizer.tokenize(sentence)
        token_count = len(tokenized)

        if current_tokens + token_count > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = token_count
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_tokens += token_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def reformat_dataframe(df: pd.DataFrame, tokenizer, max_length: int) -> pd.DataFrame:
    """Reformat dataframe by splitting long texts into smaller chunks."""
    result_rows = []

    for _, row in df.iterrows():
        text = str(row.get('text1', row.get('text', '')))
        if not text or text == 'nan':
            continue

        tokenized = tokenizer.tokenize(text)
        if len(tokenized) > max_length:
            chunks = split_string_by_sentences(text, tokenizer, max_length)
            for chunk in chunks:
                result_rows.append({'text': chunk})
        else:
            result_rows.append({'text': text})

    return pd.DataFrame(result_rows)


# =============================================================================
# Global State for Gradio App
# =============================================================================

class AppState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all state to initial values."""
        self.model: Optional[SentenceTransformer] = None
        self.model_name: str = ""
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.unique_texts: List[str] = []
        self.unique_embeddings: Optional[np.ndarray] = None
        self.technique = None
        self.technique_name: str = ""
        self.dedup_mapping: Dict[int, List[int]] = {}
        self.is_ready: bool = False

app_state = AppState()


# =============================================================================
# Core Functions
# =============================================================================

# Mapping of model choices to model IDs
MODEL_ID_MAP = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "e5-base-v2": "intfloat/e5-base-v2",
}

def load_model(model_choice: str) -> SentenceTransformer:
    """Load sentence transformer model."""
    try:
        model_id = MODEL_ID_MAP[model_choice]
    except KeyError:
        raise ValueError(f"Unknown model_choice: {model_choice}")
    return SentenceTransformer(model_id, device=device)


def process_dataset(file, model_choice: str, progress=gr.Progress()):
    """Load and process uploaded dataset."""
    if file is None:
        return "❌ Please upload an xlsx file.", None, ""
    
    # Reset state when loading new dataset (keep model if same)
    old_model = app_state.model if app_state.model_name == model_choice else None
    old_model_name = app_state.model_name if app_state.model_name == model_choice else ""
    app_state.reset()
    app_state.model = old_model
    app_state.model_name = old_model_name
    
    progress(0, desc="Loading model...")
    
    # Load model
    if app_state.model_name != model_choice:
        app_state.model = load_model(model_choice)
        app_state.model_name = model_choice
    
    tokenizer = app_state.model.tokenizer
    max_length = tokenizer.model_max_length
    
    progress(0.1, desc="Reading Excel file...")
    
    # Read Excel file
    try:
        df = pd.read_excel(file.name)
    except Exception as e:
        return f"❌ Error reading file: {str(e)}", None, ""
    
    # Find text column
    text_col = None
    for col in ['text1', 'text', 'content', 'Text', 'Content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        return f"❌ Could not find text column. Available columns: {list(df.columns)}", None, ""
    
    # Rename to standardize
    df = df.rename(columns={text_col: 'text1'})
    
    progress(0.2, desc="Reformatting texts...")
    
    # Reformat
    reformatted_df = reformat_dataframe(df, tokenizer, max_length)
    app_state.texts = reformatted_df['text'].tolist()
    
    progress(0.4, desc="Creating embeddings...")
    
    # Create embeddings
    app_state.embeddings = app_state.model.encode(
        app_state.texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype("float32")
    
    progress(1.0, desc="Done!")
    
    summary = f"""✅ **Dataset Loaded Successfully**

**Model:** {model_choice}
**Max Token Length:** {max_length}
**Original rows:** {len(df)}
**After reformatting:** {len(app_state.texts)} texts
**Embedding dimension:** {app_state.embeddings.shape[1]}
**Device:** {device}
"""
    
    # Preview
    preview_df = pd.DataFrame({
        'Index': range(min(10, len(app_state.texts))),
        'Text (preview)': [t[:100] + '...' if len(t) > 100 else t for t in app_state.texts[:10]]
    })
    
    return summary, preview_df, f"Loaded {len(app_state.texts)} texts"


def run_deduplication(method: str, threshold: float, progress=gr.Progress()):
    """Run deduplication with selected method."""
    if app_state.embeddings is None or len(app_state.texts) == 0:
        return "❌ Please load a dataset first.", None, ""
    
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / 1024 / 1024
    
    progress(0, desc=f"Initializing {method}...")
    
    # Create technique
    if method == "FAISS LSH":
        try:
            import faiss
        except ImportError:
            return "❌ FAISS not installed. Run: pip install faiss-cpu", None, ""
        technique = FaissLSHTechnique(n_bits=256)
    elif method == "SimHash":
        technique = SimHashTechnique(
            n_bits=256,
            n_bands=32,
            min_match_bands=3,  # Optimal from benchmark
            max_candidates=500,
        )
    elif method == "MinHash Embed":
        technique = MinHashEmbedTechnique(n_hashes=256, topk_dims=256, n_bands=64, min_match_bands=2)
    else:  # MinHash Shingle (text-based)
        technique = MinHashShingleTechnique(
            name="minhash_shingle",
            num_perm=256,
            ngram_size=1,
            threshold=0.45,
            min_band_matches=1,
            use_stopwords=True,
            stopwords=None,
        )
    
    progress(0.2, desc="Building index...")
    start_time = time.time()
    
    # MinHash Shingle works on text, others work on embeddings
    if method == "MinHash Shingle":
        technique.fit(app_state.texts)
    else:
        technique.fit(app_state.embeddings)
    
    progress(0.4, desc="Finding duplicates...")
    
    # Deduplication
    seen = set()
    mapping = {}
    unique_ids = []
    n_docs = len(app_state.texts)
    
    for i in range(n_docs):
        if i in seen:
            continue
        
        if i % 100 == 0:
            progress(0.4 + 0.5 * (i / n_docs), desc=f"Processing {i}/{n_docs}...")
        
        # MinHash Shingle uses text, others use embeddings
        if method == "MinHash Shingle":
            query = app_state.texts[i]
            neighbors = technique.top_k(query, k=100)
        else:
            query_vec = app_state.embeddings[i]
            neighbors = technique.top_k(query_vec, k=100)
        
        group = [i]
        for j, score in neighbors:
            if j == i:
                continue
            if score >= threshold:
                if j not in seen:
                    seen.add(j)
                    group.append(j)
        
        unique_ids.append(i)
        mapping[i] = group
    
    end_time = time.time()
    final_mem = process.memory_info().rss / 1024 / 1024
    
    # Store results
    app_state.unique_texts = [app_state.texts[i] for i in unique_ids]
    app_state.unique_embeddings = app_state.embeddings[unique_ids]
    app_state.technique = technique
    app_state.technique_name = method
    app_state.dedup_mapping = mapping
    app_state.is_ready = True
    
    # Re-fit on unique data for search
    progress(0.95, desc="Preparing search index...")
    if method == "MinHash Shingle":
        technique.fit(app_state.unique_texts)
    else:
        technique.fit(app_state.unique_embeddings)
    
    progress(1.0, desc="Done!")
    
    elapsed = end_time - start_time
    mem_used = max(0, final_mem - initial_mem)
    
    summary = f"""✅ **Deduplication Complete**

**Method:** {method}
**Threshold:** {threshold}
**Original texts:** {n_docs}
**Unique texts:** {len(unique_ids)}
**Duplicates removed:** {n_docs - len(unique_ids)}
**Reduction:** {100 * (1 - len(unique_ids) / n_docs):.1f}%

**Performance:**
- Time: {elapsed:.2f} seconds
- Memory used: {mem_used:.2f} MB
"""
    
    # Preview unique texts
    preview_data = []
    for i, uid in enumerate(unique_ids[:20]):
        group = mapping[uid]
        preview_data.append({
            'Unique ID': uid,
            'Text (preview)': app_state.texts[uid][:80] + '...' if len(app_state.texts[uid]) > 80 else app_state.texts[uid],
            'Group Size': len(group),
        })
    
    preview_df = pd.DataFrame(preview_data)
    
    return summary, preview_df, f"Ready for search ({len(unique_ids)} unique texts)"


def search_texts(query: str, top_k: int, progress=gr.Progress()):
    """Search in deduplicated dataset."""
    if not app_state.is_ready or len(app_state.unique_texts) == 0:
        return "❌ Please run deduplication first.", None, ""
    
    if not query.strip():
        return "❌ Please enter a search query.", None, ""
    
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / 1024 / 1024
    
    progress(0.2, desc="Processing query...")
    start_time = time.time()
    
    progress(0.5, desc="Searching...")
    
    # MinHash Shingle uses text directly, others use embeddings
    if app_state.technique_name == "MinHash Shingle":
        results = app_state.technique.top_k(query, k=min(top_k, len(app_state.unique_texts)))
    else:
        # Encode query for embedding-based methods
        query_vec = app_state.model.encode([query], convert_to_numpy=True)[0].astype("float32")
        results = app_state.technique.top_k(query_vec, k=min(top_k, len(app_state.unique_texts)))
    
    end_time = time.time()
    final_mem = process.memory_info().rss / 1024 / 1024
    
    progress(1.0, desc="Done!")
    
    elapsed = (end_time - start_time) * 1000  # ms
    mem_used = max(0, final_mem - initial_mem)
    
    # Build results - include unique_id for cluster lookup
    # Get mapping from unique_texts index to original unique_id
    unique_ids_list = list(app_state.dedup_mapping.keys())
    
    result_data = []
    for rank, (doc_id, score) in enumerate(results, 1):
        text = app_state.unique_texts[doc_id]
        original_unique_id = unique_ids_list[doc_id] if doc_id < len(unique_ids_list) else doc_id
        group_size = len(app_state.dedup_mapping.get(original_unique_id, []))
        result_data.append({
            'Rank': rank,
            'Unique ID': original_unique_id,
            'Group Size': group_size,
            'Score': f"{score:.4f}",
            'Text': text[:200] + '...' if len(text) > 200 else text,
        })
    
    result_df = pd.DataFrame(result_data)
    
    metrics = f"""**Search Metrics:**
- Method: {app_state.technique_name}
- Query time: {elapsed:.2f} ms
- Memory: {mem_used:.2f} MB
- Results found: {len(results)}
"""
    
    return metrics, result_df, f"Found {len(results)} results"


def export_results():
    """Export deduplicated dataset."""
    if not app_state.is_ready:
        return None, "❌ Please run deduplication first."
    
    # Create export dataframe
    export_data = []
    for i, text in enumerate(app_state.unique_texts):
        export_data.append({
            'unique_id': i,
            'text': text,
        })
    
    export_df = pd.DataFrame(export_data)
    
    # Save to temp file
    output_path = "deduplicated_output.xlsx"
    export_df.to_excel(output_path, index=False)
    
    return output_path, f"✅ Exported {len(export_data)} unique texts to {output_path}"


def view_group_members(unique_id):
    """View all texts in the same group as the selected unique text."""
    if not app_state.is_ready:
        return pd.DataFrame({'Error': ['Please run deduplication first.']})
    
    if unique_id is None:
        return pd.DataFrame({'Error': ['Please enter a Unique ID.']})
    
    unique_id = int(unique_id)
    
    if unique_id not in app_state.dedup_mapping:
        return pd.DataFrame({'Error': [f'Unique ID {unique_id} not found. Valid IDs are keys in the deduplication mapping.']})
    
    group = app_state.dedup_mapping[unique_id]
    
    group_data = []
    for idx, doc_id in enumerate(group):
        text = app_state.texts[doc_id]
        group_data.append({
            'Position': idx + 1,
            'Original Index': doc_id,
            'Is Representative': '✓' if doc_id == unique_id else '',
            'Text': text,  # Full text - click cell to see wrapped content
        })
    
    return pd.DataFrame(group_data)


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface():
    with gr.Blocks(title="Text Deduplication & Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Text Deduplication & Search Tool
        
        Upload an Excel dataset, deduplicate using FAISS, SimHash, or MinHash, then search the results.
        """)
        
        with gr.Tab("📁 1. Load Dataset"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload Excel File (.xlsx)",
                        file_types=[".xlsx"],
                    )
                    model_dropdown = gr.Dropdown(
                        choices=["all-MiniLM-L6-v2", "bge-base-en-v1.5", "e5-base-v2"],
                        value="all-MiniLM-L6-v2",
                        label="Embedding Model",
                    )
                    load_btn = gr.Button("📥 Load & Process", variant="primary")
                
                with gr.Column(scale=3):
                    load_status = gr.Markdown("Waiting for dataset...")
                    load_preview = gr.Dataframe(label="Preview (first 10 rows)")
            
            status_1 = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("⚡ 2. Deduplication"):
            with gr.Row():
                with gr.Column(scale=1):
                    method_dropdown = gr.Dropdown(
                        choices=["FAISS LSH", "SimHash", "MinHash Embed", "MinHash Shingle"],
                        value="FAISS LSH",
                        label="Deduplication Method",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Similarity Threshold",
                    )
                    dedup_btn = gr.Button("🔄 Run Deduplication", variant="primary")
                    export_btn = gr.Button("📤 Export Results")
                
                with gr.Column(scale=2):
                    dedup_status = gr.Markdown("Waiting for deduplication...")
                    dedup_preview = gr.Dataframe(label="Unique Texts Preview (click row to see group)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    group_id_input = gr.Number(label="Enter Unique ID to view group", precision=0)
                    view_group_btn = gr.Button("👁 View Group Members")
                with gr.Column(scale=2):
                    group_details = gr.Dataframe(label="Group Members (texts grouped with selected unique text)")
            
            export_file = gr.File(label="Download Deduplicated Dataset")
            export_status = gr.Textbox(label="Export Status", interactive=False)
            status_2 = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("🔎 3. Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query...",
                        lines=3,
                    )
                    topk_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Number of Results",
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary")
                
                with gr.Column(scale=2):
                    search_metrics = gr.Markdown("Enter a query and click Search")
                    search_results = gr.Dataframe(label="Search Results")
            
            status_3 = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        load_btn.click(
            fn=process_dataset,
            inputs=[file_input, model_dropdown],
            outputs=[load_status, load_preview, status_1],
        )
        
        dedup_btn.click(
            fn=run_deduplication,
            inputs=[method_dropdown, threshold_slider],
            outputs=[dedup_status, dedup_preview, status_2],
        )
        
        view_group_btn.click(
            fn=view_group_members,
            inputs=[group_id_input],
            outputs=[group_details],
        )
        
        export_btn.click(
            fn=export_results,
            inputs=[],
            outputs=[export_file, export_status],
        )
        
        search_btn.click(
            fn=search_texts,
            inputs=[search_input, topk_slider],
            outputs=[search_metrics, search_results, status_3],
        )
    
    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Ensure clean state on startup
    app_state.reset()
    demo = create_interface()
    demo.launch(share=True)
