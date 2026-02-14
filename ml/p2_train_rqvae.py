"""
Train rqvae model using embeddings. 
Combine data from multiple dataset to train rqvae.

$ python p3_train_rqvae.py
"""


import logging
import numpy as np
from flax import nnx
from quantization import rqvae, _layers
import optax
import jax
from utils import checkpointing, bagz_utils
import matplotlib.pyplot as plt
import config
import tensorflow as tf
import os

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# --- Reset logging completely ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,          # ensures info messages show
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("orbax").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def loss_fn(model, inputs):
    loss, reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm = model.compute_loss(inputs)
    return loss, (reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm)


# @nnx.jit
def train_step(model: rqvae.RQVAE, optimizer: nnx.Optimizer, inputs: jax.Array):
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, (reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm)), grads = grad_fn(model, inputs)
    optimizer.update(grads)
    return grads, loss, reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm


def save_plot(epochs, train_loss, train_reconstruction_loss, train_quantization_loss, train_usage_ratios):
    # Clear any previous plot before drawing a new one
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, linestyle='-', color='r', linewidth=1, label="Loss")
    plt.plot(epochs, train_reconstruction_loss,  linestyle='--', color='b', linewidth=1, label="Reconstruction Loss")
    plt.plot(epochs, train_quantization_loss,  linestyle='-', color='g', linewidth=1, label="Quantization Loss")
    plt.yscale('log')  # keeps x-axis linear
    plt.legend()
    plt.title("Training loss progress (log scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_usage_ratios,  linestyle='--', color='g', linewidth=1, label="train_usage_ratios")
    plt.title("Codebook usage pct")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(config.MODEL_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_rqvae_train.png"))
    plt.show()

    # Clear again after saving to ensure next call starts fresh
    plt.clf()


def get_data():

    # Load embeddings as memmap
    meta_df = bagz_utils.read_parquet(config.META_OUTSIDE_EMB)
    emb = meta_df["t5_embed"].tolist()
    
    # Convert to writable float32 arrays
    raw_item_embeddings = np.array(emb, dtype=np.float32, copy=True)

    print(f"Total items for RQVAE training: {raw_item_embeddings.shape[0]}")  

    return raw_item_embeddings



def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(config.MODEL_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_rqvae")
    
    # Load data on cpu only
    raw_item_embeddings = get_data()    # Returns a NumPy array (keep on CPU)
    raw_item_embeddings = np.array(raw_item_embeddings, dtype=np.float32)


    # Measure data variance on cpu
    data_variance = np.mean(np.var(raw_item_embeddings, axis=0)).item()
    logger.info(f"Data variance: {data_variance:.6f}")

    # Prepare dataset on cpu
    seed = np.random.randint(0, 411)
    batch_size = 2048
    dataset = (
        tf.data.Dataset.from_tensor_slices(raw_item_embeddings)
        .shuffle(buffer_size=len(raw_item_embeddings),
                 reshuffle_each_iteration=True,
                 seed=seed)
        .batch(batch_size)
        .repeat()
    )
    
    # Set hyper parameters
    hp = {
        "training": {
            "total_steps": 20_000, #20_000,
            "warmup_steps": 2_000,
        },
        "learning_rate_schedule": {
            "init_value": 0.0,
            "peak_value": 1e-3,  
            "end_value": 5e-5,
        },
        "optimizer": {
            "type": "adamw",  # or "adagrad"
            "weight_decay": 0.055,
        },
        "vqvae": {
            "num_embeddings": 256,
            "embedding_dim": 16,
            "ema_decay": 0.99,          # lower value makes code book adaptation faster, can cause instability, so training takes longer to converge
            "commitment_cost": 1.5,     # Increase commitment_cost will depress quant_loss
            "data_variance": data_variance,
        }
    }



    # Initialize the model and optimizer
    rngs = nnx.Rngs(params=0, ema=1)

    model = rqvae.RQVAE(
        input_dim=768,
        encoder_layer_dims=[512, 256, 128],
        output_dim=hp["vqvae"]["embedding_dim"],
        decoder_layer_dims=[128, 256, 512],
        quantizers=[
            _layers.VectorQuantizerEMA(hp["vqvae"]["num_embeddings"], hp["vqvae"]["embedding_dim"], rngs, decay=hp["vqvae"]["ema_decay"]),
            _layers.VectorQuantizerEMA(hp["vqvae"]["num_embeddings"], hp["vqvae"]["embedding_dim"], rngs, decay=hp["vqvae"]["ema_decay"]),
            _layers.VectorQuantizerEMA(hp["vqvae"]["num_embeddings"], hp["vqvae"]["embedding_dim"], rngs, decay=hp["vqvae"]["ema_decay"]),
            ],
        data_variance=hp["vqvae"]["data_variance"],
        commitment_cost=hp["vqvae"]["commitment_cost"],
        rngs=rngs
    )

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                                                                        init_value=hp["learning_rate_schedule"]["init_value"],         # start from zero
                                                                        peak_value=hp["learning_rate_schedule"]["peak_value"],        # max LR 
                                                                        warmup_steps=hp["training"]["warmup_steps"],    # usually 5–10% of total steps
                                                                        decay_steps=hp["training"]["total_steps"] - hp["training"]["warmup_steps"],      # total_steps - warmup_steps
                                                                        end_value=hp["learning_rate_schedule"]["end_value"]          # very low LR at end
                                                                        ), 
                                    weight_decay=hp["optimizer"]["weight_decay"]))



    train_loss = []
    train_reconstruction_loss = []
    train_quantization_loss = []
    train_usage_ratios = []

    # Initial PRNGKey
    best_reconstruction_loss = float("inf")
    iterator = iter(dataset)
    N = len(raw_item_embeddings)
    steps_per_epoch = N // batch_size
    # key = jax.random.PRNGKey(42)

    global_step = 0
    total_steps = hp["training"]["total_steps"]

    for step in range(total_steps):
        
        batch = next(iterator).numpy()  # eagerly convert TF tensor -> NumPy
        batch = jax.device_put(batch)   # Now put on GPU
        
        grads, loss, reconstruction_loss, quantization_loss, dot_reconstruction, usage_ratio, reconstruction_norm = train_step(model, optimizer, batch)
        
        train_loss.append(loss)
        train_reconstruction_loss.append(reconstruction_loss)
        train_quantization_loss.append(quantization_loss)
        train_usage_ratios.append(usage_ratio)
        
        if reconstruction_loss < best_reconstruction_loss and usage_ratio >0.7:
        # if loss < best_loss:
            checkpointing.save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                extra_params=hp
            )
            best_reconstruction_loss = reconstruction_loss
        
        global_step += 1

        # Optionally print every N steps instead of per epoch
        if step % 100 == 0:
            print(f"step={step}, train_loss={loss:.4f}, recon_loss={reconstruction_loss:.4f}, "
                f"quant_loss={quantization_loss:.4f}, mean_usage_ratio={usage_ratio:.2%},")

            # Visualize result
            steps = list(range(1, len(train_loss) + 1))
            save_plot(steps, train_loss, train_reconstruction_loss, train_quantization_loss, train_usage_ratios)


if __name__=="__main__":
    train()

    
