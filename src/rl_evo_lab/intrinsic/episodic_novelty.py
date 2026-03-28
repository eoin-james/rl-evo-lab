from __future__ import annotations

from collections import deque

import numpy as np


class EpisodicNovelty:
    """KNN novelty memory.

    Args:
        k: Number of nearest neighbours to average.
        capacity: If set, the memory becomes a rolling global buffer — the
            oldest entries are evicted automatically (O(1)) and ``reset()``
            is a no-op. If None (default), the memory is episodic and
            ``reset()`` clears it.
    """

    def __init__(self, k: int, capacity: int | None = None) -> None:
        self.k = k
        self._capacity = capacity
        self._memory: deque[np.ndarray] = deque(maxlen=capacity)

    def reset(self) -> None:
        if self._capacity is None:
            self._memory.clear()
        # Global buffers ignore reset() — they persist across episodes.

    def query(self, embedding: np.ndarray) -> float:
        """Return mean KNN distance without modifying memory."""
        if len(self._memory) < self.k:
            return 0.0
        memory = np.stack(self._memory)
        dists = np.linalg.norm(memory - embedding, axis=1)
        return float(np.partition(dists, self.k - 1)[: self.k].mean())

    def add(self, embedding: np.ndarray) -> None:
        """Insert an embedding into memory."""
        self._memory.append(embedding.copy())

    def score(self, embedding: np.ndarray) -> float:
        """Query KNN distance then add embedding to memory (episodic convenience method)."""
        novelty = self.query(embedding)
        self.add(embedding)
        return novelty
