from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from model import MonolithModel

Batch = dict[str, Any]  # keys: sparse (dict), dense (array), labels (array)


def _loss_fn(
  params: Any,
  model: MonolithModel,
  sparse: dict[str, list[int]],
  dense: jax.Array,
  labels: jax.Array,
) -> jax.Array:
  saved = model.params
  model.params = params
  logits = model(sparse, dense)
  model.params = saved
  return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def _step(
  model: MonolithModel,
  batch: Batch,
  opt_state: Any,
  optimizer: optax.GradientTransformation,
) -> tuple[Any, float]:
  """Compute loss/grads and apply a single optimizer update."""
  loss, grads = jax.value_and_grad(_loss_fn)(model.params, model, batch["sparse"], batch["dense"], batch["labels"])
  updates, opt_state = optimizer.update(grads, opt_state, model.params)
  model.params = optax.apply_updates(model.params, updates)
  return opt_state, float(loss)


def batch_train(
  model: MonolithModel,
  data: list[Batch],
  epochs: int = 5,
  lr: float = 1e-3,
) -> list[float]:
  """Standard training loop. Returns per-epoch losses."""
  optimizer = optax.adam(lr)
  opt_state = optimizer.init(model.params)
  losses: list[float] = []

  for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in data:
      opt_state, loss = _step(model, batch, opt_state, optimizer)
      epoch_loss += loss
    avg = epoch_loss / max(len(data), 1)
    losses.append(avg)
    print(f"Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

  return losses


def online_update(
  model: MonolithModel,
  batch: Batch,
  opt_state: Any,
  optimizer: optax.GradientTransformation,
) -> tuple[Any, float]:
  """Single-step online update. Returns (new_opt_state, loss)."""
  return _step(model, batch, opt_state, optimizer)
