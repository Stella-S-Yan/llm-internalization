

from flax import nnx
import jax
import jax.numpy as jnp
from typing import Sequence


def get_sharded_data(
    x: jnp.ndarray
) -> jnp.ndarray:
  """Shards the data before passing it to the model."""
  mesh= jax.make_mesh((8, 1), ('data', 'model'))
  data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))
  return jax.device_put(x, data_sharding)



def clone_model_optimizer(
    optimizer: nnx.Optimizer,
) -> tuple[nnx.Module, nnx.Optimizer]:
  model = optimizer.model
  cloned_model = nnx.clone(model)
  tx = optimizer.tx
  cloned_optimizer = nnx.Optimizer(cloned_model, tx)
  return cloned_model, cloned_optimizer


class BaseSharding:

  def __init__(self, optimizer: nnx.Optimizer):
    self.sharded_model = None
    self.sharded_optimizer = None

  def get_sharded_components(self):
    """Returns the sharded model and optimizer."""
    return self.sharded_model, self.sharded_optimizer


class DataParallelTraining(BaseSharding):

  def __init__(self, optimizer: nnx.Optimizer):
    """Sets up model and data parallelism using JAX sharding."""
    super().__init__(optimizer)
    self._setup_sharding()
    self._initialize_sharded_model_optimizer(optimizer)
    self._replicate_state()

  def _setup_sharding(self):
    """Configures JAX sharding and device mesh."""
    num_devices = jax.local_device_count()
    self.mesh = jax.make_mesh((num_devices,), ('data',))
    self.model_sharding = jax.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec()
    )
    self.data_sharding = jax.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec('data')
    )

  def _initialize_sharded_model_optimizer(self, optimizer: nnx.Optimizer):
    """Creates a deep-copied, sharded version of the model."""
    self.sharded_model, self.sharded_optimizer = clone_model_optimizer(
        optimizer
    )

  def _replicate_state(self):
    """Replicates the model and optimizer state across devices."""
    self.to_shard_state = nnx.state(
        (self.sharded_model, self.sharded_optimizer)
    )
    self.sharded_state = jax.device_put(
        self.to_shard_state, self.model_sharding
    )
    nnx.update((self.sharded_model, self.sharded_optimizer), self.sharded_state)


def dereplicate(sharding_obj: BaseSharding):
  """Consolidates the states from multiple devices into a single instance."""
  sharded_model, sharded_optimizer = sharding_obj.get_sharded_components()

  state = nnx.state((sharded_model, sharded_optimizer))
  state = jax.device_get(state)
  derep_model = nnx.clone(sharded_model)
  derep_optimizer = nnx.clone(sharded_optimizer)
  nnx.update((derep_model, derep_optimizer), state)
  return derep_model, derep_optimizer