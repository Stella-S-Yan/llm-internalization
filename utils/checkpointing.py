from flax import nnx
import orbax.checkpoint as ocp
import os
import jax
import logging

logging.basicConfig(level=logging.DEBUG)

def save_checkpoint(checkpoint_dir, step, model, optimizer, extra_params=None):
    path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
    path = os.path.join(path, str(step))

    _, state = nnx.split((model, optimizer))
    mngr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    mngr.save(path, args=ocp.args.Composite(state=ocp.args.PyTreeSave(state), metadata=ocp.args.JsonSave(extra_params)))


def load_checkpoint(checkpoint_dir):
    """This project prefers to save only one checkpoint."""
    folders = os.listdir(checkpoint_dir)
    if not folders:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # If multiple checkpoints exist, pick the first (or sort if needed)
    load_step = folders[0]

    mngr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    path = os.path.join(checkpoint_dir, load_step)
    # restored = mngr.restore(path)
    restored = mngr.restore(
        path,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeRestore(),  # state will be loaded as a pytree
            metadata=ocp.args.JsonRestore()            # tell Orbax how to load metadata
        )
    )
    return restored


def save_linen_checkpoint(checkpoint_dir, step, state):
    checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
    # checkpoint_dir = os.path.join(checkpoint_dir, str(step))

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    # Manager handles multiple checkpoints (e.g. keep only last N)
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)
    manager.save(step, state)


def load_linen_checkpoint(checkpoint_dir, step):
    checkpoint_dir = str(checkpoint_dir)  # ensure string for Orbax

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist")

    # Orbax requires a "checkpointer" (like a driver)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    # Manager handles multiple checkpoints (keep only last N)
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=False)
    manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)

    # Restore specific step
    state = manager.restore(step=step)

    return state
