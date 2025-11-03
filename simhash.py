import hashlib
import numpy as np

class SimHash:
    def __init__(self, num_bits=64):
        """
        num_bits: length of the SimHash fingerprint (usually 64 or 128)
        """
        self.num_bits = num_bits

    def _hash(self, token):
        """Hash a token into an integer using SHA-1."""
        return int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16)

    def compute(self, tokens):
        """
        Compute the SimHash value for a list of tokens (e.g., words in a document).
        Returns a binary string of length num_bits.
        """
        v = np.zeros(self.num_bits)

        for token in tokens:
            h = self._hash(token)
            for i in range(self.num_bits):
                bitmask = 1 << i
                if h & bitmask:
                    v[i] += 1   # weight could be TF-IDF or frequency
                else:
                    v[i] -= 1

        # Build final hash: positive means bit 1, negative means bit 0
        fingerprint = 0
        for i in range(self.num_bits):
            if v[i] >= 0:
                fingerprint |= 1 << i
        return fingerprint

    def hamming_distance(self, hash1, hash2):
        """Compute Hamming distance between two SimHash fingerprints."""
        x = hash1 ^ hash2
        return bin(x).count('1')


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    simhash = SimHash(num_bits=64)

    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A quick brown fox jumped over a lazy dog"

    # Simple tokenization by splitting on whitespace
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()

    hash1 = simhash.compute(tokens1)
    hash2 = simhash.compute(tokens2)

    print(f"SimHash 1: {bin(hash1)}")
    print(f"SimHash 2: {bin(hash2)}")
    print(f"Hamming distance: {simhash.hamming_distance(hash1, hash2)}")
