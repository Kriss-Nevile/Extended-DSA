

import os
import time
from typing import Dict, List, Tuple, Union, Set, Optional
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil
import torch
import hashlib


# :contentReference[oaicite:1]{index=1}

# =============================================================================
# 0. Cấu hình đường dẫn & thiết bị
# =============================================================================

PROJECT_ROOT = "C:\\My_data\\Work_spaces\\251\\Extend_DSA\\v5"

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
EVAL_DIR = os.path.join(PROJECT_ROOT, "Evaluation")
EMB_DIR  = os.path.join(PROJECT_ROOT, "Embeddings")  # NEW

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(EMB_DIR,  exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

try:
    import faiss
    print("FAISS đã cài sẵn:", faiss.__version__)
except ImportError:
    # !pip install -q faiss-cpu Not working in pure script
    import faiss
    print("Đã cài faiss-cpu:", faiss.__version__)

"""### Base method

"""

# =============================
# 1. Hashing: SimHash & MinHash Encoder
# =============================

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

        neighbors = []
        for d, idx in zip(distances[0], indices[0]):
            neighbors.append((int(idx), float(d)))
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





    # 2) Khởi tạo techniques
simhash_tech = SimHashTechnique(
    n_bits=256,
    n_bands=32,
    min_match_bands=3,  # Optimal từ benchmark
    max_candidates=500,
)

minhash_tech = MinHashShingleTechnique(
    name="minhash_shingle",
    num_perm=256,
    ngram_size=1,
    threshold=0.45,
    min_band_matches=1,
    use_stopwords=True,
    stopwords=None,
)


faiss_lsh_tech = FaissLSHTechnique(n_bits=256)

# (optional) baseline exact
# faiss_exact_tech = FaissExactTechnique()
