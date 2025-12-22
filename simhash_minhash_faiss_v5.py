"""
Module chứa các hàm và class cho SimHash, MinHash, và FAISS
Được trích xuất từ Deduplicate_text_v2.ipynb
"""

import re
import hashlib
from collections import defaultdict
from typing import List, Tuple, Union, Set, Optional, Dict

import numpy as np
import faiss


# =============================
# Base Classes
# =============================

class HashEncoder:
    """Base class cho các encoder"""
    def fit(self, docs: np.ndarray):
        raise NotImplementedError

    def encode(self, docs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BaseIndex:
    """Base class cho các index"""
    def build(self, docs: np.ndarray, signatures: np.ndarray = None):
        raise NotImplementedError

    def query(self, query_vec: np.ndarray, top_k: int = 10, encoder: HashEncoder = None):
        raise NotImplementedError


class BaseSearchTechnique:
    """Interface chung: nhận embedding vector, trả về (doc_id, score)."""
    def __init__(self, name: str):
        self.name = name

    def fit(self, docs: np.ndarray) -> None:
        raise NotImplementedError

    def top_k(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError


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
    LSH Index cho SimHash sử dụng banding và sort-based bucketing.
    Tối ưu hóa RAM bằng cách dùng flat arrays thay vì dictionaries.
    """
    def __init__(self, n_bits, n_bands=32, min_match_bands=1, max_rerank=200):
        self.n_bits = n_bits
        self.n_bands = n_bands
        self.min_match_bands = min_match_bands
        self.max_rerank = max_rerank

        # Kiểm tra tính hợp lệ
        self.n_bytes = n_bits // 8
        if self.n_bytes * 8 != n_bits:
            raise ValueError("n_bits phải chia hết cho 8.")

        self.bytes_per_band = self.n_bytes // n_bands

        # Cấu trúc dữ liệu dạng mảng phẳng (Flat Arrays)
        self.bands_data = []
        self.docs_norm = None
        self.sig_packed = None

        # Popcount table cho Hamming distance
        self._popcnt = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    def build(self, docs: np.ndarray, packed_signatures: np.ndarray):
        """
        Dùng sort để gom nhóm bucket, loại bỏ hoàn toàn Python Dictionary và List.
        """
        self.sig_packed = packed_signatures
        N = docs.shape[0]

        # Chuẩn hoá docs
        norms = np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9
        self.docs_norm = (docs / norms).astype(np.float32)

        # Xây dựng Index theo từng Band bằng Sort
        self.bands_data = []
        dtype_void = np.dtype((np.void, self.bytes_per_band))

        for b in range(self.n_bands):
            start = b * self.bytes_per_band
            end = (b + 1) * self.bytes_per_band

            # Lấy key của band hiện tại
            keys = self.sig_packed[:, start:end]

            # Chuyển sang dạng void để sort
            keys_void = np.ascontiguousarray(keys).view(dtype_void).ravel()

            # Argsort: Lấy thứ tự index sao cho keys tăng dần
            sort_idx = np.argsort(keys_void)

            # Sắp xếp keys và doc_ids theo thứ tự đó
            sorted_keys = keys_void[sort_idx]
            sorted_doc_ids = sort_idx.astype(np.int32)

            # Tìm các boundary của bucket
            changes = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1], [True]))
            start_indices = np.where(changes)[0]

            # Chỉ lưu unique keys
            unique_keys = sorted_keys[start_indices[:-1]]

            self.bands_data.append({
                'keys': unique_keys,
                'starts': start_indices,
                'doc_ids': sort_idx
            })

        print(f"[SimHashLSHIndex] Built {self.n_bands} bands using Sort-based LSH. RAM optimized.")

    def query(self, query_vec, encoder, top_k=10):
        """Query với binary search và cosine reranking"""
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

            # Binary search tìm bucket
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

        # 4. Hamming Filter
        candidate_sigs = self.sig_packed[candidate_ids]
        xor_result = np.bitwise_xor(candidate_sigs, q_packed)
        hamming_dists = self._popcnt[xor_result].sum(axis=1)

        if self.max_rerank and len(candidate_ids) > self.max_rerank:
            top_hamming_idx = np.argpartition(hamming_dists, self.max_rerank)[:self.max_rerank]
            candidate_ids = candidate_ids[top_hamming_idx]

        # 5. Cosine Rerank
        q_vec = np.asarray(query_vec, dtype=np.float32).flatten()
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        target_vecs = self.docs_norm[candidate_ids]
        scores = target_vecs @ q_norm

        # 6. Final Top K
        if len(scores) > top_k:
            best_idxs = np.argpartition(-scores, top_k)[:top_k]
            final_idxs = best_idxs[np.argsort(-scores[best_idxs])]
        else:
            final_idxs = np.argsort(-scores)

        return [(int(candidate_ids[i]), float(scores[i])) for i in final_idxs]


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
        self.buckets = None
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

        # buckets[b]: dict[band_key(tuple)] -> list[doc_id]
        self.buckets = [defaultdict(list) for _ in range(self.n_bands)]
        for doc_id in range(N):
            sig = signatures[doc_id]
            for b in range(self.n_bands):
                start = b * self.band_width
                end = start + self.band_width
                band = tuple(sig[start:end].tolist())
                self.buckets[b][band].append(doc_id)

    def _collect_candidates(self, q_sig: np.ndarray) -> np.ndarray:
        candidates = []
        for b in range(self.n_bands):
            start = b * self.band_width
            end = start + self.band_width
            key = tuple(q_sig[start:end].tolist())
            if key in self.buckets[b]:
                candidates.extend(self.buckets[b][key])

        if not candidates:
            return np.array([], dtype=int)

        cand_arr = np.array(candidates, dtype=int)
        unique_ids, counts = np.unique(cand_arr, return_counts=True)
        mask = counts >= self.min_match_bands
        final = unique_ids[mask]
        if final.size == 0:
            final = unique_ids
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
        num_perm: int = 128,
        ngram_size: int = 1,
        seed: int = 42,
        use_stopwords: bool = False,
        stopwords: Optional[Set[str]] = None,
    ):
        self.num_perm = int(num_perm)
        self.ngram_size = int(ngram_size)
        self.seed = int(seed)

        self.use_stopwords = bool(use_stopwords)
        self.stopwords = stopwords if stopwords is not None else set()

        # Prime modulus for universal hashing
        self.mersenne_prime = (1 << 61) - 1
        p = self.mersenne_prime

        rng = np.random.default_rng(self.seed)
        self.a = rng.integers(1, p, size=(self.num_perm,), dtype=np.uint64)
        self.b = rng.integers(0, p, size=(self.num_perm,), dtype=np.uint64)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

        if self.use_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    @staticmethod
    def _hash64(s: str) -> np.uint64:
        """Stable 64-bit hash of a string using blake2b."""
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
        return np.uint64(int.from_bytes(h.digest(), "little", signed=False))

    def _get_shingles(self, text: str) -> np.ndarray:
        """
        Return unique 64-bit shingle hashes (uint64).
        """
        tokens = self._tokenize(text)
        n = len(tokens)

        if n == 0:
            return np.array([0], dtype=np.uint64)

        k = min(self.ngram_size, n)
        shingles: List[np.uint64] = []

        for i in range(n - k + 1):
            ngram = " ".join(tokens[i : i + k])
            shingles.append(self._hash64(ngram))

        return np.unique(np.array(shingles, dtype=np.uint64))

    def encode(self, docs: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """
        Return MinHash signatures: shape (N, num_perm), dtype uint64
        """
        if isinstance(docs, str):
            docs_list = [docs]
        else:
            docs_list = list(docs)

        N = len(docs_list)
        sigs = np.empty((N, self.num_perm), dtype=np.uint64)

        p = self.mersenne_prime
        a = self.a.reshape(1, -1)  # (1, P)
        b = self.b.reshape(1, -1)  # (1, P)

        for i, doc in enumerate(docs_list):
            sh = self._get_shingles(doc)  # (S,)

            # Reduce x into modulo field
            sh_col = (sh % p).reshape(-1, 1)  # (S,1)

            # (S,1)*(1,P) -> (S,P); then take min over S
            hv = (sh_col * a + b) % p
            sigs[i] = hv.min(axis=0)

        return sigs


# =============================
# MinHash Shingle Index
# =============================

class MinHashShingleIndex:
    """
    LSH Index cho MinHash signatures với:
    - Stable band hash (blake2b)
    - Auto choose (b,r) near threshold OR allow manual (b,r)
    - min_band_matches to reduce false positives
    """
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        b: Optional[int] = None,
        r: Optional[int] = None,
        min_band_matches: int = 1,
    ):
        self.num_perm = int(num_perm)
        self.threshold = float(threshold)
        self.min_band_matches = int(min_band_matches)

        if b is None or r is None:
            self.b, self.r, self.used_perm = self._choose_br(self.num_perm, self.threshold)
        else:
            self.b, self.r = int(b), int(r)
            self.used_perm = self.b * self.r
            if self.used_perm > self.num_perm:
                raise ValueError("Invalid (b,r): b*r must be <= num_perm")

        self.buckets: Dict[int, List[int]] = defaultdict(list)
        self.signatures: Optional[np.ndarray] = None

        print(f"LSH Config: Bands={self.b}, Rows={self.r} (Using {self.used_perm}/{self.num_perm} perms)")
        print(f"threshold={self.threshold} | min_band_matches={self.min_band_matches}")

    @staticmethod
    def _choose_br(num_perm: int, threshold: float) -> Tuple[int, int, int]:
        """
        Choose (b,r) with b*r <= num_perm such that (1/b)^(1/r) ~ threshold.
        """
        best = None  # (error, -used, b, r)
        for r in range(1, num_perm + 1):
            b = num_perm // r
            if b <= 0:
                continue
            used = b * r
            t_est = (1.0 / b) ** (1.0 / r)
            err = abs(t_est - threshold)

            key = (err, -used, b, r)
            if best is None or key < best:
                best = key

        _, _, b, r = best
        return b, r, b * r

    @staticmethod
    def _stable_band_hash(band_vec: np.ndarray, band_idx: int) -> int:
        """Stable across runs. Convert band to bytes then hash with blake2b."""
        h = hashlib.blake2b(digest_size=8)
        h.update(band_vec.tobytes())
        h.update(band_idx.to_bytes(2, "little", signed=False))
        return int.from_bytes(h.digest(), "little", signed=False)

    def build(self, signatures: np.ndarray) -> None:
        """
        signatures: shape (N, num_perm), dtype uint64
        """
        if signatures.shape[1] < self.used_perm:
            raise ValueError(
                f"Signatures have fewer permutations than required by (b,r). "
                f"Need >= {self.used_perm}, got {signatures.shape[1]}."
            )

        # Reset buckets if rebuild
        self.buckets.clear()

        # Store only the used prefix
        self.signatures = np.ascontiguousarray(signatures[:, : self.used_perm], dtype=np.uint64)

        N = self.signatures.shape[0]
        for doc_idx in range(N):
            sig = self.signatures[doc_idx]
            for band_idx in range(self.b):
                start = band_idx * self.r
                end = start + self.r
                band_vec = sig[start:end]
                key = self._stable_band_hash(band_vec, band_idx)
                self.buckets[key].append(doc_idx)

    def query(self, query_sig: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Returns list of (doc_idx, estimated_jaccard) sorted desc.
        """
        if self.signatures is None:
            raise RuntimeError("Index is empty. Call build() first.")

        q = np.asarray(query_sig, dtype=np.uint64)[: self.used_perm]

        hit_count: Dict[int, int] = defaultdict(int)
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_vec = q[start:end]
            key = self._stable_band_hash(band_vec, band_idx)
            for doc_idx in self.buckets.get(key, []):
                hit_count[doc_idx] += 1

        candidates = [i for i, c in hit_count.items() if c >= self.min_band_matches]
        if not candidates:
            return []

        cand_sigs = self.signatures[candidates]  # (C, used_perm)
        matches = (cand_sigs == q)
        scores = matches.mean(axis=1)

        results = [(candidates[i], float(scores[i])) for i in range(len(candidates)) if scores[i] > 0.0]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


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


# =============================
# Complete Search Techniques
# =============================

class SimHashTechnique(BaseSearchTechnique):
    """
    SimHash + LSH trên embedding.
    """
    def __init__(
        self,
        n_bits: int = 256,
        n_bands: int = 32,
        min_match_bands: int = 5,
        max_rerank: int = 500,
    ):
        super().__init__(name="simhash")
        self.n_bits = n_bits
        self.n_bands = n_bands
        self.min_match_bands = min_match_bands
        self.max_rerank = max_rerank
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

        # 2) Build LSH index
        index = SimHashLSHIndex(
            n_bits=self.n_bits,
            n_bands=self.n_bands,
            min_match_bands=self.min_match_bands,
            max_rerank=self.max_rerank,
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


class MinHashShingleTechnique(BaseSearchTechnique):
    """
    MinHash + LSH trên text shingles.
    """
    def __init__(
        self,
        name: str = "minhash_shingle",
        num_perm: int = 128,
        ngram_size: int = 1,
        threshold: float = 0.85,
        min_band_matches: int = 1,
        b: Optional[int] = None,
        r: Optional[int] = None,
        seed: int = 42,
        use_stopwords: bool = False,
        stopwords: Optional[Set[str]] = None,
    ):
        super().__init__(name)

        self.encoder = MinHashShingleEncoder(
            num_perm=num_perm,
            ngram_size=ngram_size,
            seed=seed,
            use_stopwords=use_stopwords,
            stopwords=stopwords,
        )
        self.index = MinHashShingleIndex(
            num_perm=num_perm,
            threshold=threshold,
            b=b,
            r=r,
            min_band_matches=min_band_matches,
        )

        self.docs = None

    def fit(self, docs: np.ndarray) -> None:
        print(f"[{self.name}] Encoding {len(docs)} documents...")
        self.docs = docs
        signatures = self.encoder.encode(docs)
        self.index.build(signatures)
        print(f"[{self.name}] Index built.")

    def top_k(self, query: Union[str, np.ndarray], k: int) -> List[Tuple[int, float]]:
        if isinstance(query, str):
            query_sig = self.encoder.encode([query])[0]
        else:
            query_sig = query

        return self.index.query(query_sig, top_k=k)


def demo_create_techniques():
    """
    Ví dụ cách tạo và sử dụng các techniques.
    """
    print("=" * 80)
    print("DEMO: CÁCH TẠO CÁC TECHNIQUES")
    print("=" * 80)
    
    # ==========================================
    # 1. FAISS Exact (Ground Truth)
    # ==========================================
    print("\n1. FAISS Exact Technique:")
    print("-" * 40)
    faiss_exact = FaissExactTechnique()
    print(f"   Tên: {faiss_exact.name}")
    print("   Mục đích: Tìm kiếm chính xác (Ground Truth)")
    print("   Sử dụng: faiss_exact.fit(docs)")
    print("            results = faiss_exact.top_k(query_vec, k=10)")
    
    # ==========================================
    # 2. FAISS LSH
    # ==========================================
    print("\n2. FAISS LSH Technique:")
    print("-" * 40)
    faiss_lsh = FaissLSHTechnique(n_bits=128)
    print(f"   Tên: {faiss_lsh.name}")
    print(f"   Số bits: {faiss_lsh.n_bits}")
    print("   Mục đích: LSH implementation của FAISS")
    print("   Sử dụng: faiss_lsh.fit(docs)")
    print("            results = faiss_lsh.top_k(query_vec, k=10)")
    
    # ==========================================
    # 3. SimHash Technique
    # ==========================================
    print("\n3. SimHash Technique:")
    print("-" * 40)
    simhash = SimHashTechnique(
        n_bits=256,
        n_bands=32,
        min_match_bands=5,
        max_rerank=500,
    )
    print(f"   Tên: {simhash.name}")
    print(f"   Số bits: {simhash.n_bits}")
    print(f"   Số bands: {simhash.n_bands}")
    print(f"   Min match bands: {simhash.min_match_bands}")
    print(f"   Max rerank: {simhash.max_rerank}")
    print("   Mục đích: SimHash với LSH trên embedding vectors")
    print("   Sử dụng: simhash.fit(docs)")
    print("            results = simhash.top_k(query_vec, k=10)")
    
    # ==========================================
    # 4. MinHash Embed Technique
    # ==========================================
    print("\n4. MinHash Embed Technique:")
    print("-" * 40)
    minhash_embed = MinHashEmbedTechnique(
        n_hashes=256,
        topk_dims=256,
        n_bands=64,
        min_match_bands=2,
    )
    print(f"   Tên: {minhash_embed.name}")
    print(f"   Số hashes: {minhash_embed.n_hashes}")
    print(f"   Top-k dims: {minhash_embed.topk_dims}")
    print(f"   Số bands: {minhash_embed.n_bands}")
    print(f"   Min match bands: {minhash_embed.min_match_bands}")
    print("   Mục đích: MinHash trên embedding vectors (top-k dimensions)")
    print("   Sử dụng: minhash_embed.fit(docs)")
    print("            results = minhash_embed.top_k(query_vec, k=10)")
    
    # ==========================================
    # 5. MinHash Shingle Technique
    # ==========================================
    print("\n5. MinHash Shingle Technique:")
    print("-" * 40)
    minhash_shingle = MinHashShingleTechnique(
        name="minhash_shingle",
        num_perm=128,
        ngram_size=1,
        threshold=0.45,
        min_band_matches=1,
        seed=42,
        use_stopwords=False,
    )
    print(f"   Tên: {minhash_shingle.name}")
    print(f"   Số permutations: {minhash_shingle.encoder.num_perm}")
    print(f"   N-gram size: {minhash_shingle.encoder.ngram_size}")
    print(f"   Threshold: {minhash_shingle.index.threshold}")
    print(f"   Bands: {minhash_shingle.index.b}")
    print(f"   Rows per band: {minhash_shingle.index.r}")
    print(f"   Min band matches: {minhash_shingle.index.min_band_matches}")
    print("   Mục đích: MinHash trên text shingles (cho near-duplicate detection)")
    print("   Sử dụng: minhash_shingle.fit(text_array)")
    print("            results = minhash_shingle.top_k(query_text, k=10)")
    
    # ==========================================
    # 6. Ví dụ với các cấu hình khác
    # ==========================================
    print("\n" + "=" * 80)
    print("CÁC CẤU HÌNH KHÁC:")
    print("=" * 80)
    
    print("\n6a. MinHash Shingle với bigram:")
    minhash_bigram = MinHashShingleTechnique(
        name="minhash_bigram",
        num_perm=256,
        ngram_size=2,  # Bigram
        threshold=0.75,
        min_band_matches=2,
    )
    print(f"    num_perm={minhash_bigram.encoder.num_perm}, ngram_size={minhash_bigram.encoder.ngram_size}")
    
    print("\n6b. SimHash với cấu hình nhẹ hơn:")
    simhash_light = SimHashTechnique(
        n_bits=128,
        n_bands=16,
        min_match_bands=3,
        max_rerank=200,
    )
    print(f"    n_bits={simhash_light.n_bits}, n_bands={simhash_light.n_bands}")
    
    print("\n6c. MinHash Embed với cấu hình mạnh hơn:")
    minhash_strong = MinHashEmbedTechnique(
        n_hashes=512,
        topk_dims=512,
        n_bands=128,
        min_match_bands=3,
    )
    print(f"    n_hashes={minhash_strong.n_hashes}, n_bands={minhash_strong.n_bands}")
    
    print("\n" + "=" * 80)
    print("LƯU Ý:")
    print("=" * 80)
    print("1. Techniques hoạt động với embedding vectors PHẢI fit với docs array")
    print("   (SimHash, MinHashEmbed, FAISS)")
    print("2. MinHashShingle hoạt động với TEXT strings")
    print("3. Sau khi fit(), dùng top_k() để tìm kiếm")
    print("4. Kết quả trả về: List[(doc_id, score)]")
    print("=" * 80)


if __name__ == "__main__":
    print("Module chứa các class và hàm SimHash, MinHash, FAISS")
    print("Import module này để sử dụng các class:")
    print("  - SimHashEncoder, SimHashLSHIndex, SimHashTechnique")
    print("  - MinHashEmbedEncoder, MinHashEmbedLSHIndex, MinHashEmbedTechnique")
    print("  - MinHashShingleEncoder, MinHashShingleIndex, MinHashShingleTechnique")
    print("  - FaissExactTechnique, FaissLSHTechnique")
    print("\n")
    
    # Chạy demo
    demo_create_techniques()
