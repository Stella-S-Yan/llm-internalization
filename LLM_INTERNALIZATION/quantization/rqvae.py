import jax
import jax.numpy as jnp
import flax.nnx as nnx
from LLM_INTERNALIZATION.quantization._layers import Encoder, Decoder, ResidualVectorQuantizer

from typing import List

def ste(x, x_hat):
    """Straight through estimator
    A trick to allow gradients to flow through non-differentialble operations like 
    quantization. 
    """
    return x + jax.lax.stop_gradient(x_hat - x)
    
class RQVAE(nnx.Module):
    """
    num_hiddens: input size to the residual stack.
    num_residual_layers: how many residual blocks the encoder has.
    num_residual_hiddens: hidden layer size within each residual block
    num_embeddings: number of codebook entries
    embedding_dim: codebook entry dimensionality
    depth: the number of residual quantization steps. How many times the input is passed through the quantization process, typically [1~3]
    decay: 
    commitment_cost: 
    """
    def __init__(self, 
                 input_dim: int,
                 encoder_layer_dims: list[int],
                 output_dim: int,
                 decoder_layer_dims: list[int],
                 quantizers: List[nnx.Module],
                 data_variance: float,
                 commitment_cost: float,
                 rngs: nnx.Rngs):
        
        self.data_variance = data_variance
        self.commitment_cost = commitment_cost
        self._encoder = Encoder(in_dim=input_dim,
                                layer_dims = encoder_layer_dims,
                                out_dim=output_dim,
                                rngs=rngs)
        
        self.residual_quantizers = ResidualVectorQuantizer(quantizers=quantizers)
        
        self._decoder = Decoder(in_dim=output_dim,
                                layer_dims=decoder_layer_dims,
                                out_dim=input_dim,
                                rngs=rngs)
        
    def __call__(self, inputs, training=False):
        x = self._encoder(inputs)
        
        # start residual quantization
        if training:
            quantized_latent, codebook_indices, usage_ratios, quantization_loss = self.residual_quantizers(x, training)
            reconstructions = self._decoder(quantized_latent)
            return reconstructions, codebook_indices, usage_ratios, quantization_loss
        else:
            quantized_latent, codebook_indices, usage_ratios = self.residual_quantizers(x, training)
            reconstructions = self._decoder(quantized_latent)
            return reconstructions, codebook_indices, usage_ratios
        
    
    def compute_loss(self, inputs):
        reconstructions, codebook_indices, usage_ratios, quantization_loss = self(inputs, True)
        reconstruction_loss = jnp.mean((reconstructions - inputs) ** 2) / self.data_variance
        loss = reconstruction_loss + self.commitment_cost * quantization_loss
        dot_reconstruction = jnp.sum(inputs * reconstructions, axis=1)
        # l2 norm of each reconstruction acorss its dimensions. 
        reconstruction_norm = jnp.linalg.norm(reconstructions, axis=1)
        # mean usage_ratios
        mean_usage_ratios = jnp.mean(usage_ratios, axis=0)
        
        return loss, reconstruction_loss, quantization_loss, dot_reconstruction, mean_usage_ratios, reconstruction_norm
    

    def decode_from_codebook_indices(self, codebook_indices):
        # Get quantized vectors from each codebook level
        quantized_vectors = []
        for i, idx in enumerate(codebook_indices):
            embedding = self.residual_quantizers.quantizers[i].embedding # shape [K, D]
            quantized_vector = embedding[idx]  # shape: [D]
            quantized_vectors.append(quantized_vector)

        # Sum residuals to get the final quantized latent
        quantized_latent = sum(quantized_vectors)  # shape: [D], where D is latent dim (e.g., 256)

        # Decode to get reconstruction in original space (e.g., 768-dim)
        reconstruction = self._decoder(quantized_latent)
        return reconstruction

    # def decode_from_codebook_indices(self, codebook_indices):
    # """
    # Decode reconstructions from stored codebook indices.

    # Args:
    #     codebook_indices: (L, B, D)
    #       L = # residual quantization levels
    #       B = batch size
    #       D = latent dim (e.g. 768)

    # Returns:
    #     reconstruction: (B, input_dim)
    # """
    # L = len(self.residual_quantizers.quantizers)
    # if codebook_indices.shape[0] != L:
    #     raise ValueError(
    #         f"Expected {L} levels, got {codebook_indices.shape[0]} in codebook_indices"
    #     )

    def decode_from_codebook_indices(self, codebook_indices):
        """
        Decode reconstructions from stored codebook indices.

        Args:
            codebook_indices: (L, B, D)
            L = # residual quantization levels
            B = batch size
            D = latent dim (e.g. 768)

        Returns:
            reconstruction: (B, input_dim)
        """
        L = len(self.residual_quantizers.quantizers)
        if codebook_indices.shape[0] != L:
            raise ValueError(
                f"Expected {L} levels, got {codebook_indices.shape[0]} in codebook_indices"
            )

        z_levels = []
        for l, q in enumerate(self.residual_quantizers.quantizers):
            idx = codebook_indices[l]  # shape (B, D)
            # q.embedding: (K, D), idx: (B, D)
            # gather -> (B, D, D), then take diagonal across embedding_dim axis
            # But here we need per-dimension lookup. Each dimension d chooses one codeword from K.
            # So we gather row-wise for each d.
            e = q.embedding  # (K, D)
            # Use advanced indexing: for each b,d, lookup embedding[idx[b,d], d]
            b_idx, d_idx = jnp.meshgrid(
                jnp.arange(idx.shape[0]), jnp.arange(idx.shape[1]), indexing="ij"
            )
            z_l = e[idx, d_idx]  # (B, D)
            z_levels.append(z_l)

        # Residual sum across levels -> (B, D)
        z_q = sum(z_levels)

        # Decode back to input space -> (B, input_dim)
        return self._decoder(z_q)
