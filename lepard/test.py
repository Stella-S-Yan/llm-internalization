from utils import load_model
import config
import logging
from flax import nnx
import optax
from utils import checkpointing
from lepard.ddp import DataParallelTraining, get_sharded_data, dereplicate
from utils.checkpointing import load_checkpoint
import jax
import numpy as np

logger = logging.getLogger(__name__)

def show_leaves(tree, name, n=20):
    leaves = jax.tree_util.tree_leaves(tree)
    print(f"{name}: {len(leaves)} leaves")
    for i, leaf in enumerate(leaves[:n]):
        try:
            shape = np.shape(leaf)
        except Exception:
            shape = type(leaf)
        print(f"  leaf[{i}] shape/type: {shape}")


class MLP(nnx.Module):
  def __init__(self, din, dmid, dout, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    return self.linear2(nnx.relu(self.linear1(x)))

model = MLP(1, 64, 1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-2))


# Save original model
fpath = "/usr/local/google/home/stellasyan/Documents/llm_internalization/data/original_model_checkpoint"
checkpointing.save_checkpoint(
    checkpoint_dir=fpath,
    step=1,
    model=model,
    optimizer=optimizer,
    extra_params={}
)

ori_restored = load_checkpoint(checkpoint_dir=fpath)
graphdef, x = nnx.split((model, optimizer))
restored_model, restored_optimizer = nnx.merge(graphdef, ori_restored["state"])

print("is nnx.State?", isinstance(ori_restored["state"], nnx.State))

show_leaves(ori_restored["state"], "restored_state")

# Sharding
fpath = "/usr/local/google/home/stellasyan/Documents/llm_internalization/data/shard_model_checkpoint"
ddp = DataParallelTraining(optimizer)
sharded_model, sharded_optimizer = ddp.get_sharded_components()

derep_model, derep_optimizer = dereplicate(ddp)
checkpointing.save_checkpoint(
    checkpoint_dir=fpath,
    step=1,
    model=derep_model,
    optimizer=derep_optimizer,
    extra_params={}
)

shard_restored = load_checkpoint(checkpoint_dir=fpath)

print("is nnx.State?", isinstance(shard_restored["state"], nnx.State))

show_leaves(shard_restored["state"], "restored_state")




shard_graphdef, _ = nnx.split((model, optimizer))
# shard_graphdef, _ = nnx.split((sharded_model, sharded_optimizer))
# restored_model, restored_optimizer = nnx.merge(shard_graphdef, ori_restored["state"]) # Works
restored_model, restored_optimizer = nnx.merge(shard_graphdef, shard_restored["state"]) # failed


# print(graphdef)
# print("--------------------------------")
# print(shard_graphdef)


# print("=======================")
# print(ori_restored["state"])
# print("--------------------------------")
# print(shard_restored["state"])

# print("here")