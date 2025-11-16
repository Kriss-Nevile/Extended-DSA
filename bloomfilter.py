import math
import hashlib
import bitarray 


"""The bloom filter is a space-efficient probabilistic data structure 
    used to test whether an element is a member of a set.
    It returns either "possibly in set" or "definitely not in set".

    This is useful for our text deduplication tasks, where we want to quickly check 
    if a document has already been seen.

    It is not a hash algorithm itself, but rather a data structure that uses multiple hash functions

    For a normal hash algorithm, we need to build a separate search method. With a bloom filter,
    we can directly check membership.
"""


class BloomFilter:
    def __init__(self, n_items, false_positive_rate):
        """
        Initialize Bloom Filter parameters.
        n_items: expected number of elements to store
        false_positive_rate: acceptable false positive probability (e.g., 0.01)
        """
        self.m = self._get_size(n_items, false_positive_rate)  
        """This is the total number of bits in the bit array"""
        self.k = self._get_hash_count(self.m, n_items)
        self.bit_array = bitarray.bitarray(self.m)
        self.bit_array.setall(0)
        print(f"Initialized BloomFilter with {self.m} bits and {self.k} hash functions")

    def _get_size(self, n, p):
        """Compute size of bit array (m)"""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    def _get_hash_count(self, m, n):
        """Compute number of hash functions (k)"""
        k = (m / n) * math.log(2)
        return int(k)

    def _hashes(self, item):
        """Generate k different hash values for an item"""
        hashes = []
        item_bytes = item.encode('utf-8')
        for i in range(self.k):
            hash_val = int(hashlib.sha256(item_bytes + i.to_bytes(2, 'little')).hexdigest(), 16)
            hashes.append(hash_val % self.m)
        return hashes

    def add(self, item):
        """Insert an item into the filter"""
        for h in self._hashes(item):
            self.bit_array[h] = True

    def check(self, item):
        """Check if item is possibly in set"""
        return all(self.bit_array[h] for h in self._hashes(item))


if __name__ == "__main__":
    bf = BloomFilter(n_items=1000, false_positive_rate=0.01)

    bf.add("apple")
    bf.add("banana")

    print("apple:", bf.check("apple"))      # True (probably)
    print("banana:", bf.check("banana"))    # True (probably)
    print("cherry:", bf.check("cherry"))    # False (definitely not)