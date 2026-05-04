import jax
import jax.numpy as jnp
from flax import nnx

from typing import List
 
class Encoder(nnx.Module):
    def __init__(self, in_dim, layer_dims, out_dim, rngs: nnx.Rngs):
        layer_sizes = [in_dim] + layer_dims + [out_dim]
        self.num_layers = len(layer_sizes) - 1
        
        print(layer_sizes)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nnx.Linear(in_features=layer_sizes[i],
                                            out_features=layer_sizes[i + 1],
                                            kernel_init=nnx.initializers.glorot_uniform(),
                                            bias_init = nnx.initializers.zeros,
                                            rngs=rngs
                                            ))
        
    def __call__(self, x):
        for i in range(self.num_layers-1): # No activation for the last layer
            x = self.layers[i](x)
            x = nnx.relu(x)
        return self.layers[-1](x) 
    

class Decoder(nnx.Module):
    def __init__(self, in_dim, layer_dims, out_dim, rngs: nnx.Rngs):
        layer_sizes = [in_dim] + layer_dims + [out_dim]
        self.num_layers = len(layer_sizes) - 1
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nnx.Linear(in_features=layer_sizes[i],
                                     out_features=layer_sizes[i + 1],
                                     kernel_init=nnx.initializers.glorot_uniform(),
                                     bias_init = nnx.initializers.zeros,
                                     rngs=rngs
                                     ))
        
    def __call__(self, x):
        for i in range(self.num_layers-1): # No activation for the last layer
            x = self.layers[i](x)
            x = nnx.relu(x)
        x = self.layers[-1](x)
        return x 

        
class VectorQuantizerEMA(nnx.Embed):
    def __init__(self, num_embeddings: int, embedding_dim: int, rngs: nnx.Rngs, decay=0.99, eps=1e-5):
        super().__init__(num_embeddings=num_embeddings, 
                        features=embedding_dim, 
                        embedding_init= jax.nn.initializers.normal(1.0),
                        rngs=rngs)
        
        self._decay = decay
        self._eps = eps
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self._ema_cluster_size = nnx.Variable(jnp.zeros(num_embeddings))
        # Holding ema of centroids
        self._ema_w = nnx.Variable(nnx.initializers.normal(1.0)(rngs.ema(), (self.num_embeddings, self.embedding_dim)))
        
    def _codebook_usage(self, encoding):
        usage_counts = jnp.sum(encoding, axis=0)
        num_used_entries = jnp.count_nonzero(usage_counts)  
        usage_ratio = num_used_entries / self.num_embeddings
        
        return usage_ratio
        
    def _compute_distances(self, inputs):
        codebook = self.embedding.raw_value
        distances = (
                    jnp.sum(inputs ** 2, axis=1, keepdims=True)  # (N, 1)
                    + jnp.sum(codebook ** 2, axis=1)  # (n_embed,)
                    - 2.0 * jnp.matmul(inputs, codebook.T)  # (N, n_embed)
                    )
        return distances
    
    def _update_codebook(self, inputs, encoding):
        # Use EMA to update the codebook. Note, first update count, then centroids
        """ 
        To avoid the case where clusters with fewer assignments dominate the update, you normalize the cluster sizes. 
        Smaller clusters will have their influence reduced by the normalized term, because the ratio
        cluster_size_ema[i] / n will be small compared to larger clusters. Larger cluster will have their influcence 
        increased or prserved. Preventing the model from collapsing to a small subset of embeddings
        """
        cluster_size = self._ema_cluster_size.raw_value * self._decay + (1.0 - self._decay) * jnp.sum(encoding, axis=0)
        # Laplace smoothing of the cluster size
        n = jnp.sum(cluster_size)
        self._ema_cluster_size.raw_value = ((cluster_size + self._eps) / (n + self.num_embeddings * self._eps) * n )

        # Sums all input vectors assigned to each codebook entry.
        total_assignment_sums = jnp.matmul(encoding.T, inputs) # (N, n_embed) * (N, embed_dim) -> (n_embed, embed_dim)
        # Update the centroids
        self._ema_w.raw_value = self._ema_w.raw_value * self._decay + (1.0 - self._decay) * total_assignment_sums
        # Update codebook (embeddings)
        self.embedding.raw_value = self._ema_w.raw_value / (jnp.expand_dims(self._ema_cluster_size.raw_value, axis=1) + self._eps)

    def _compute_loss(self, inputs, quantized):
        '''
        In VQ-EMA, there is no need to compute embedding_loss, unlike standard VQ. 
        * embedding_loss: Ensures that the embedding vectors (centroids) move toward the input data points.
        * encoder_loss: Forces the encoded inputs to move toward the selected quantized embeddings
        In VQ-EMA, embeddings are updated by EMA instead of receiving gradient updates, so no need for embedding_loss. 
        But the encoder network still should learn to output representations that are close to their assigned quantized values,
        so still need encoder_loss. 
        '''
        quantization_loss = jnp.mean((inputs - quantized) ** 2) # encoder/commitment loss. Ensure encoder output (inputs) stay close to quantized representation.
        return quantization_loss
     
    def __call__(self, inputs, training: bool = False):
        
        # Compute distances
        distances = self._compute_distances(inputs)
        
        # Encoding
        emb_idxs = jnp.argmin(distances, axis=-1) # (N, )
        encoding = nnx.one_hot(emb_idxs, num_classes=self.num_embeddings) # (N, n_embed)
        
        # Quantize
        
        # Use one-hot encoding to retrieve the corresponding codebook entry = quantization
        codebook = self.embedding.raw_value
        quantized = jnp.matmul(encoding, codebook) # (N, embed_dim) 
        
        # Unflatten: reshape to original input shape
        quantized = quantized.reshape(inputs.shape)
        
        # Monitor codebook usage
        usage_ratio = self._codebook_usage(encoding)
        
        # Use EMA to update the embedding vectors  
        if training:
            
            self._update_codebook(inputs, encoding)
            quantization_loss = self._compute_loss(inputs, quantized)
            quantized_flow = inputs + jax.lax.stop_gradient(quantized - inputs)
            return quantized_flow, encoding, usage_ratio, quantization_loss
        else:
            return quantized, encoding, usage_ratio


class ResidualVectorQuantizer(nnx.Module):
    def __init__(self, quantizers: List[nnx.Module]):
        self.quantizers = quantizers
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False):
        quantized = []
        encodings = []
        usage_ratios = []
        residual = inputs
        total_quantization_loss = 0.0
        
        for idx, quantizer in enumerate(self.quantizers):
            if training: 
                current_quantized, current_encoding, usage_ratio, quantization_loss = quantizer(residual, training)
                total_quantization_loss += quantization_loss
            else:
                current_quantized, current_encoding, usage_ratio = quantizer(residual, training)
            quantized.append(current_quantized)
            residual -= current_quantized
            encodings.append(current_encoding)
            usage_ratios.append(usage_ratio)
        
        if training:
            return jnp.sum(jnp.stack(quantized, axis=0), axis=0), jnp.stack(encodings, axis=0), jnp.stack(usage_ratios, axis=0), total_quantization_loss
        else:
            return jnp.sum(jnp.stack(quantized, axis=0), axis=0), jnp.stack(encodings, axis=0), jnp.stack(usage_ratios, axis=0)