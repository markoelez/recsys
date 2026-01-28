from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from embedding import CuckooHashEmbeddingTable


@dataclass
class SparseFeatureConfig:
  name: str
  capacity: int
  embed_dim: int
  min_frequency: int = 2
  ttl: float | None = None


class MLP(nn.Module):
  hidden_sizes: list[int]

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    for size in self.hidden_sizes:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    x = nn.Dense(1)(x)
    return x.squeeze(-1)


class MonolithModel:
  """Sparse + dense → MLP → CTR prediction.

  The cuckoo embedding tables are stateful Python objects that live outside the
  Flax parameter tree.  The MLP weights are standard Flax params.
  """

  def __init__(
    self,
    sparse_configs: list[SparseFeatureConfig],
    dense_dim: int,
    hidden_sizes: list[int] | None = None,
  ) -> None:
    self.sparse_configs = sparse_configs
    self.dense_dim = dense_dim
    self.hidden_sizes = hidden_sizes or [128, 64]

    self.tables: dict[str, CuckooHashEmbeddingTable] = {
      cfg.name: CuckooHashEmbeddingTable(
        capacity=cfg.capacity,
        embed_dim=cfg.embed_dim,
        min_frequency=cfg.min_frequency,
        ttl=cfg.ttl,
      )
      for cfg in sparse_configs
    }

    self.mlp = MLP(hidden_sizes=self.hidden_sizes)

    total_sparse_dim = sum(c.embed_dim for c in sparse_configs)
    dummy_input = jnp.zeros(total_sparse_dim + dense_dim)
    self.params = self.mlp.init(jax.random.PRNGKey(0), dummy_input)

  def __call__(
    self,
    sparse_features: dict[str, list[int]],
    dense_features: jax.Array,
  ) -> jax.Array:
    """Forward pass. Returns logits (pre-sigmoid) of shape (batch,)."""
    parts: list[jax.Array] = []
    for cfg in self.sparse_configs:
      ids = sparse_features[cfg.name]
      parts.append(self.tables[cfg.name].lookup(ids))

    sparse_concat = jnp.concatenate(parts, axis=-1)  # (batch, total_sparse_dim)
    x = jnp.concatenate([sparse_concat, dense_features], axis=-1)
    logits = jax.vmap(lambda row: self.mlp.apply(self.params, row))(x)
    return logits

  def predict(
    self,
    sparse_features: dict[str, list[int]],
    dense_features: jax.Array,
  ) -> jax.Array:
    return jax.nn.sigmoid(self(sparse_features, dense_features))
