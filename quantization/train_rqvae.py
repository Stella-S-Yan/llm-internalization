"""
Train rqvae model using embeddings. 
"""


import pandas as pd
import logging
import numpy as np
import jax.numpy as jnp
from flax import nnx
from quantization import rqvae, _layers
import optax
import jax
from utils import checkpointing
import matplotlib.pyplot as plt
import config
import tensorflow as tf
from utils import bagz_utils
import os

logging.getLogger("orbax").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



def loss_fn(model, inputs):
    loss, reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm = model.compute_loss(inputs)
    return loss, (reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm)


# @nnx.jit
def train_step(model: rqvae.RQVAE, optimizer: nnx.ModelAndOptimizer, inputs: jax.Array):
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, (reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm)), grads = grad_fn(model, inputs)
    optimizer.update(grads)
    return grads, loss, reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm


def save_plot(epochs, train_loss, train_reconstruction_loss, train_quantization_loss, train_usage_ratios):
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
    plt.savefig(config.TRAIN_LOSS_PLOT)
    plt.show()


def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    meta_df = bagz_utils.read_parquet(config.META_W_EMBEDDING)

    raw_item_embeddings = meta_df['embedding'].tolist()
    checkpoint_dir=config.RQVAE_CHECKPOINT_DIR

    # Ensure all arrays are writable
    raw_item_embeddings = [np.array(emb, dtype=np.float32, copy=True) for emb in raw_item_embeddings]

    # Measure data variance
    all_data = jnp.array(raw_item_embeddings)
    data_variance = jnp.var(all_data)
    data_variance = data_variance.item()
    logger.info(f"data variance: {data_variance}")

    # Prepare dataset
    batch_size = 2048
    dataset = tf.data.Dataset.from_tensor_slices(all_data)
    dataset = dataset.shuffle(buffer_size=len(all_data)).batch(batch_size).repeat() # Shuffle whole dataset

    # Set hyper parameters
    hp = {
        "training": {
            "total_steps": 10_000, #20_000,
            "warmup_steps": 2_000,
        },
        "learning_rate_schedule": {
            "init_value": 0.0,
            "peak_value": 1e-3,  # learning_rate
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
            "commitment_cost": 0.3,
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

    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(learning_rate=optax.schedules.warmup_cosine_decay_schedule(
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
    best_loss = 10.0
    iterator = iter(dataset)
    steps_per_epoch = len(all_data) // batch_size

    for epoch in range(hp["training"]["total_steps"]):
        for step in range(steps_per_epoch):
            batch = next(iterator)
            batch = jax.device_put(jnp.array(batch))   # convert TF tensor → JAX array
            grads, loss, reconstruction_loss, quantization_loss, dot_reconstruction, usage_ratio, reconstruction_norm = train_step(model, optimizer, batch)
            train_loss.append(loss)
            train_reconstruction_loss.append(reconstruction_loss)
            train_quantization_loss.append(quantization_loss)
            train_usage_ratios.append(usage_ratio)
            
            

            if loss < best_loss and usage_ratio >0.8:
            # if loss < best_loss:
                checkpointing.save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=epoch,
                    model=model,
                    optimizer=optimizer,
                    extra_params=hp
                )
                best_loss = loss
        
        if epoch % 10 == 0:
            print(f"epoch={epoch}, train_loss={loss:.4f}, recon_loss={reconstruction_loss:.4f}, quant_loss={quantization_loss:.4f}, mean_usage_ratio={usage_ratio:.2%}" )


    # Visualize result
    epochs = list(range(1, len(train_loss) + 1))  # Epoch numbers
    save_plot(epochs, train_loss, train_reconstruction_loss, train_quantization_loss, train_usage_ratios)

if __name__=="__main__":
    train()

    
