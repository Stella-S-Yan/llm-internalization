
from flax import nnx
import optax
import logging

import config
from utils.checkpointing import load_checkpoint
from quantization.rqvae import RQVAE
from quantization._layers import VectorQuantizerEMA

logging.basicConfig(level=logging.DEBUG)


def load_rqvae(checkpoint_dir=None):
    
    # Load model checkpoint
    restored = load_checkpoint(checkpoint_dir=checkpoint_dir)

    hp = restored["metadata"]


    rngs = nnx.Rngs(params=0, ema=1)

    abstract_model = RQVAE(
        input_dim=768,
        encoder_layer_dims=[512, 256, 128],
        output_dim=hp["vqvae"]["embedding_dim"],
        decoder_layer_dims=[128, 256, 512],
        quantizers=[
            VectorQuantizerEMA(hp["vqvae"]["num_embeddings"], hp["vqvae"]["embedding_dim"], rngs, decay=hp["vqvae"]["ema_decay"]),
            VectorQuantizerEMA(hp["vqvae"]["num_embeddings"], hp["vqvae"]["embedding_dim"], rngs, decay=hp["vqvae"]["ema_decay"]),
            VectorQuantizerEMA(hp["vqvae"]["num_embeddings"], hp["vqvae"]["embedding_dim"], rngs, decay=hp["vqvae"]["ema_decay"]),
            ],
        data_variance=hp["vqvae"]["data_variance"],
        commitment_cost=hp["vqvae"]["commitment_cost"],
        rngs=rngs
    )

    abstract_optimizer = nnx.Optimizer(abstract_model, optax.adamw(learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                                                                        init_value=hp["learning_rate_schedule"]["init_value"],         # start from zero
                                                                        peak_value=hp["learning_rate_schedule"]["peak_value"],        # max LR 
                                                                        warmup_steps=hp["training"]["warmup_steps"],    # usually 5–10% of total steps
                                                                        decay_steps=hp["training"]["total_steps"] - hp["training"]["warmup_steps"],      # total_steps - warmup_steps
                                                                        end_value=hp["learning_rate_schedule"]["end_value"]          # very low LR at end
                                                                        ), 
                                    weight_decay=hp["optimizer"]["weight_decay"]))

    graphdef, x = nnx.split((abstract_model, abstract_optimizer))

    restored_model, restored_optimizer = nnx.merge(graphdef, restored["state"])
    logging.info("RQVAE model restored.")
    
    return restored_model, restored_optimizer


