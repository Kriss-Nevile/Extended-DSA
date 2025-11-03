import re
import unicodedata
import random
from typing import List, Tuple, Dict, Set


class MinHasher:
    """
    A minimal and educational implementation of classic MinHash for text similarity.
    Steps:
      1. Normalize text and generate character n-gram shingles.
      2. Hash each shingle using 64-bit FNV-1a.
      3. Apply k random hash functions h_i(x) = (A[i] * x + B[i]) mod p.
      4. Construct a MinHash signature of k elements per document.
      5. Estimate Jaccard similarity using the proportion of matching positions.

    Example usage:
        m = MinHasher(n=5, k=128, seed=42)
        shingles_map, signatures, params = m.build_signatures(corpus)
        pairs = m.pairwise_minhash(corpus, show_true=True)
    """

    # ---------------------- Initialization ----------------------
    def __init__(self, n: int = 5, k: int = 128, seed: int = 42) -> None:
        """
        n    : shingle size (character n-grams)
        k    : number of hash functions
        seed : random seed for reproducibility
        """
        self.n = n
        self.k = k
        self.seed = seed
        # A large Mersenne prime ~ 2^61 - 1 (used as modulus)
        self.p = 2305843009213693951
        self.A, self.B = self._init_hash_family(k, seed, self.p)

    # ---------------------- Normalization & Shingling ----------------------
    @staticmethod
    def normalize_text(s: str) -> str:
        """Unicode normalization + lowercase + collapse whitespace."""
        s = unicodedata.normalize("NFC", s)
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def fnv1a_64(s: str) -> int:
        """Compute 64-bit FNV-1a hash of a UTF-8 string, returning signed-like int."""
        data = s.encode("utf-8")
        FNV_OFFSET_BASIS = 0xcbf29ce484222325
        FNV_PRIME = 0x100000001b3
        h = FNV_OFFSET_BASIS
        for b in data:
            h ^= b
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        # Convert to signed-like int for consistency
        if h & (1 << 63):
            h = -((~h & 0xFFFFFFFFFFFFFFFF) + 1)
        return h

    def char_shingles(self, text: str) -> Set[int]:
        """
        Convert text into a set of hashed character n-grams.
        If text is shorter than n, use the whole string as one shingle.
        """
        t = self.normalize_text(text)
        if len(t) < self.n:
            return {self.fnv1a_64(t)}
        S: Set[int] = set()
        for i in range(len(t) - self.n + 1):
            g = t[i:i + self.n]
            S.add(self.fnv1a_64(g))
        return S

    # ---------------------- Hash Family ----------------------
    @staticmethod
    def _init_hash_family(k: int, seed: int, p: int) -> Tuple[List[int], List[int]]:
        """Initialize k linear hash functions: h_i(x) = (A[i]*x + B[i]) mod p."""
        rng = random.Random(seed)
        A = [rng.randrange(1, p - 1) for _ in range(k)]
        B = [rng.randrange(0, p - 1) for _ in range(k)]
        return A, B

    def _h_i(self, x: int, i: int) -> int:
        """Compute the i-th hash function value."""
        if x < 0:
            x = -x
        return (self.A[i] * x + self.B[i]) % self.p

    # ---------------------- MinHash Signature ----------------------
    def minhash_signature(self, S: Set[int]) -> List[int]:
        """
        Compute MinHash signature for set S:
        sig[i] = min_{x in S} h_i(x)
        """
        INF = self.p
        sig = [INF] * self.k
        for x in S:
            for i in range(self.k):
                v = self._h_i(x, i)
                if v < sig[i]:
                    sig[i] = v
        return sig

    # ---------------------- Jaccard Similarity ----------------------
    @staticmethod
    def jaccard_estimate(sigA: List[int], sigB: List[int]) -> float:
        """Estimate Jaccard similarity from two signatures."""
        assert len(sigA) == len(sigB)
        k = len(sigA)
        matches = sum(1 for i in range(k) if sigA[i] == sigB[i])
        return matches / k

    @staticmethod
    def jaccard_true(SA: Set[int], SB: Set[int]) -> float:
        """Compute the true Jaccard similarity of two shingle sets."""
        if not SA and not SB:
            return 1.0
        inter = len(SA & SB)
        uni = len(SA | SB)
        return inter / max(1, uni)

    # ---------------------- Signature Generation ----------------------
    def build_signatures(
        self, corpus: Dict
    ) -> Tuple[Dict, Dict, Tuple[List[int], List[int], int]]:
        """
        Build MinHash signatures for an entire corpus.
        corpus: {doc_id -> text}

        Returns:
            - shingles_map: {id -> set of shingles}
            - signatures  : {id -> MinHash signature (list[int])}
            - hash_params : (A, B, p)
        """
        shingles_map: Dict = {}
        signatures: Dict = {}
        for doc_id, text in corpus.items():
            S = self.char_shingles(text)
            sig = self.minhash_signature(S)
            shingles_map[doc_id] = S
            signatures[doc_id] = sig
        return shingles_map, signatures, (self.A, self.B, self.p)

    # ---------------------- Pairwise Comparison ----------------------
    def pairwise_minhash(
        self,
        corpus: Dict,
        show_true: bool = True
    ) -> List[Tuple]:
        """
        Compute pairwise MinHash Jaccard estimates between all pairs (i < j).
        If show_true=True, also include the true Jaccard similarity.
        Results are sorted descending by estimated similarity.
        """
        shingles_map, signatures, _ = self.build_signatures(corpus)
        ids = list(corpus.keys())
        results = []
        for a_idx in range(len(ids)):
            for b_idx in range(a_idx + 1, len(ids)):
                ia, ib = ids[a_idx], ids[b_idx]
                jhat = self.jaccard_estimate(signatures[ia], signatures[ib])
                if show_true:
                    jtrue = self.jaccard_true(shingles_map[ia], shingles_map[ib])
                    results.append((ia, ib, jhat, jtrue))
                else:
                    results.append((ia, ib, jhat))
        results.sort(key=lambda x: -x[2])  # most similar pairs first
        return results


# ---------------------- TIỆN ÍCH IN ẤN ----------------------
def _print_results(title: str, results: List[Tuple], top: int = None) -> None:
    print("="*80)
    print(title)
    print("-"*80)
    header = "DocA DocB | Jaccard_est  Jaccard_true"
    print(header)
    print("-"*len(header))
    rows = results if top is None else results[:top]
    for row in rows:
        if len(row) == 4:
            a, b, jhat, jtrue = row
            print(f"{a:>4} {b:>4} |    {jhat:>8.3f}      {jtrue:>8.3f}")
        else:
            a, b, jhat = row
            print(f"{a:>4} {b:>4} |    {jhat:>8.3f}")
    print()


# ---------------------- DEMO 1: BASIC ----------------------
def demo1_basic():
    """
    Mức cơ bản: 3 tài liệu, 2 cái khá giống nhau, 1 cái khác hẳn.
    Thông số phổ biến: n=3 (char-3gram), k=128, seed=7
    """
    corpus = {
        1: "Minhash is a technique for estimating Jaccard similarity between sets.",
        2: "Jaccard similarity between sets can be estimated using MinHash signatures.",
        3: "Totally different content with no overlap at all."
    }
    n, k, seed = 3, 128, 7
    m = MinHasher(n=n, k=k, seed=seed)
    results = m.pairwise_minhash(corpus, show_true=True)
    _print_results(f"[DEMO 1] Basic — n={n}, k={k}, seed={seed}", results)


# ---------------------- DEMO 2: VIETNAMESE & NORMALIZATION ----------------------
def demo2_vietnamese():
    """
    Mức trung bình: Văn bản tiếng Việt với dấu, viết hoa, nhiều khoảng trắng/ ký tự.
    Mục tiêu: thấy hiệu quả chuẩn hoá Unicode + lower + gom khoảng trắng.
    """
    corpus = {
        1: "TÔI   THÍCH học   máy và    xử  lý NGÔN ngữ     tự NHIÊN.",
        2: "Tôi thích học máy & xử lý ngôn ngữ tự nhiên!",
        3: "Hôm nay trời đẹp, tôi đi học.",
        4: "Xử lý ngôn ngữ tự nhiên là một lĩnh vực của học máy."
    }
    n, k, seed = 3, 128, 42
    m = MinHasher(n=n, k=k, seed=seed)
    results = m.pairwise_minhash(corpus, show_true=True)
    _print_results(f"[DEMO 2] Vietnamese & Normalization — n={n}, k={k}, seed={seed}", results)


# ---------------------- DEMO 3: SCALING & THRESHOLDING ----------------------
def demo3_scaling():
    """
    Mức nâng cao nhẹ: 6 tài liệu, có 2 cụm chủ đề khác nhau.
    - So sánh hiệu ứng số hàm băm k (64 vs 256).
    - Lọc cặp nghi ngờ trùng/giống (Jhat >= 0.5).
    """
    corpus = {
        # Cụm A: về MinHash/Jaccard
        1: "MinHash is used to estimate Jaccard similarity efficiently.",
        2: "Estimating Jaccard similarity can be done with MinHash signatures.",
        3: "We compute MinHash signatures to approximate the Jaccard index.",
        # Cụm B: khác chủ đề (nấu ăn)
        4: "I love cooking pasta with tomato sauce and fresh basil.",
        5: "Fresh tomatoes and basil are great for making pasta sauce.",
        6: "Baking bread requires patience and attention to fermentation."
    }

    # Run 1: k=64
    n, k1, seed = 3, 64, 999
    m1 = MinHasher(n=n, k=k1, seed=seed)
    res1 = m1.pairwise_minhash(corpus, show_true=True)
    _print_results(f"[DEMO 3] Scaling — n={n}, k={k1}, seed={seed} (Top 10)", res1, top=10)

    # Run 2: k=256
    k2 = 256
    m2 = MinHasher(n=n, k=k2, seed=seed)
    res2 = m2.pairwise_minhash(corpus, show_true=True)
    _print_results(f"[DEMO 3] Scaling — n={n}, k={k2}, seed={seed} (Top 10)", res2, top=10)

    # # Lọc cặp có Jhat >= 0.5 ở cấu hình k=256 (độ tin cậy cao hơn)
    # threshold = 0.5
    # filtered = [row for row in res2 if row[2] >= threshold]
    # _print_results(f"[DEMO 3] Candidate near-duplicates (k={k2}), Jhat >= {threshold}", filtered)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    # Chạy lần lượt 3 demo. Bạn có thể comment/bật từng cái tuỳ ý.
    demo1_basic()
    demo2_vietnamese()
    demo3_scaling()