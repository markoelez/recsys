"""Minimal demo: synthetic CTR data → batch train → online updates."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import optax
from sklearn.metrics import roc_auc_score

from model import MonolithModel, SparseFeatureConfig
from train import batch_train, online_update


def make_synthetic_data(
  n: int,
  n_users: int = 500,
  n_items: int = 1000,
  dense_dim: int = 4,
  seed: int = 42,
) -> list[dict]:
  rng = np.random.RandomState(seed)
  user_ids = rng.randint(0, n_users, size=n).tolist()
  item_ids = rng.randint(0, n_items, size=n).tolist()
  dense = rng.randn(n, dense_dim).astype(np.float32)
  labels = rng.randint(0, 2, size=n).astype(np.float32)
  return [
    {
      "sparse": {"user": user_ids, "item": item_ids},
      "dense": jnp.array(dense),
      "labels": jnp.array(labels),
    }
  ]


def main() -> None:
  sparse_configs = [
    SparseFeatureConfig(name="user", capacity=2048, embed_dim=8, min_frequency=1),
    SparseFeatureConfig(name="item", capacity=4096, embed_dim=8, min_frequency=1),
  ]
  dense_dim = 4

  model = MonolithModel(sparse_configs, dense_dim=dense_dim, hidden_sizes=[64, 32])

  # --- batch training ---
  print("=== Batch Training ===")
  train_data = make_synthetic_data(1024, dense_dim=dense_dim)
  batch_train(model, train_data, epochs=100, lr=1e-3)

  # --- online training ---
  print("\n=== Online Training ===")
  optimizer = optax.adam(1e-4)
  opt_state = optimizer.init(model.params)

  online_data = make_synthetic_data(256, dense_dim=dense_dim, seed=99)
  for i, batch in enumerate(online_data):
    opt_state, loss = online_update(model, batch, opt_state, optimizer)
    preds = model.predict(batch["sparse"], batch["dense"])
    auc = roc_auc_score(np.array(batch["labels"]), np.array(preds))
    print(f"Online step {i + 1}  loss={loss:.4f}  AUC={auc:.4f}")


if __name__ == "__main__":
  main()
