from __future__ import annotations

import time

import jax
import jax.numpy as jnp


class CuckooHashEmbeddingTable:
  """Collisionless embedding table using cuckoo hashing.

  Two internal tables with independent hash functions guarantee a unique slot
  per feature ID.  Supports frequency filtering (only materialise an embedding
  once the ID has been seen >= *min_frequency* times) and TTL-based expiry.
  """

  def __init__(
    self,
    capacity: int,
    embed_dim: int,
    max_kicks: int = 500,
    min_frequency: int = 2,
    ttl: float | None = None,
    init_scale: float = 0.01,
  ) -> None:
    self.capacity = capacity
    self.embed_dim = embed_dim
    self.max_kicks = max_kicks
    self.min_frequency = min_frequency
    self.ttl = ttl
    self.init_scale = init_scale

    # slot -> (feature_id, embedding)
    self.table0: dict[int, tuple[int, jax.Array]] = {}
    self.table1: dict[int, tuple[int, jax.Array]] = {}

    # frequency counter and last-access timestamps
    self.freq: dict[int, int] = {}
    self.last_access: dict[int, float] = {}

  # -- hash functions -------------------------------------------------------

  def _hash0(self, fid: int) -> int:
    h = ((fid * 0x9E3779B1) & 0xFFFFFFFF) >> 16
    return h % self.capacity

  def _hash1(self, fid: int) -> int:
    h = ((fid * 0x85EBCA6B + 0xC2B2AE35) & 0xFFFFFFFF) >> 16
    return h % self.capacity

  # -- core operations ------------------------------------------------------

  def _make_embedding(self, fid: int) -> jax.Array:
    key = jax.random.PRNGKey(fid)
    return jax.random.normal(key, (self.embed_dim,)) * self.init_scale

  def _find(self, fid: int) -> jax.Array | None:
    slot0 = self._hash0(fid)
    entry = self.table0.get(slot0)
    if entry is not None and entry[0] == fid:
      return entry[1]
    slot1 = self._hash1(fid)
    entry = self.table1.get(slot1)
    if entry is not None and entry[0] == fid:
      return entry[1]
    return None

  def _insert(self, fid: int) -> None:
    cur_id, cur_emb = fid, self._make_embedding(fid)

    for kick in range(self.max_kicks):
      # try table0
      slot0 = self._hash0(cur_id)
      if slot0 not in self.table0:
        self.table0[slot0] = (cur_id, cur_emb)
        return
      # evict from table0, place current there
      evicted = self.table0[slot0]
      self.table0[slot0] = (cur_id, cur_emb)
      cur_id, cur_emb = evicted

      # try table1
      slot1 = self._hash1(cur_id)
      if slot1 not in self.table1:
        self.table1[slot1] = (cur_id, cur_emb)
        return
      evicted = self.table1[slot1]
      self.table1[slot1] = (cur_id, cur_emb)
      cur_id, cur_emb = evicted

    raise RuntimeError(f"Cuckoo insert failed after {self.max_kicks} kicks — consider increasing capacity")

  def lookup(self, ids: list[int]) -> jax.Array:
    """Return stacked embeddings for *ids*.

    IDs that haven't yet met the frequency threshold get a zero vector.
    """
    now = time.time()
    results: list[jax.Array] = []
    zero = jnp.zeros(self.embed_dim)

    for fid in ids:
      self.freq[fid] = self.freq.get(fid, 0) + 1
      self.last_access[fid] = now

      emb = self._find(fid)
      if emb is not None:
        results.append(emb)
        continue

      if self.freq[fid] >= self.min_frequency:
        self._insert(fid)
        results.append(self._find(fid))  # type: ignore[arg-type]
      else:
        results.append(zero)

    return jnp.stack(results)

  def evict_expired(self) -> int:
    """Remove entries whose last access exceeds the TTL. Returns count evicted."""
    if self.ttl is None:
      return 0
    now = time.time()
    evicted = 0
    for table in (self.table0, self.table1):
      expired_slots = [slot for slot, (fid, _) in table.items() if now - self.last_access.get(fid, 0) > self.ttl]
      for slot in expired_slots:
        fid = table[slot][0]
        del table[slot]
        self.freq.pop(fid, None)
        self.last_access.pop(fid, None)
        evicted += 1
    return evicted

  def __len__(self) -> int:
    return len(self.table0) + len(self.table1)

  def __contains__(self, fid: int) -> bool:
    return self._find(fid) is not None
