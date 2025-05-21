import torch
import torch.nn as nn
from typing import Dict
import numpy as np
import torch.nn.functional as F

# Configuration for GPT
GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "num_heads" : 12,
    "num_layers": 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}
class GPTModel(nn.Module):
    """
    GPT model to predict the next word in a sentence, given the words.
    Reference:
        [Book]
            https://www.manning.com/books/build-a-large-language-model-from-scratch
        [GitHub]
            https://github.com/rasbt/LLMs-from-scratch

    Consists of 
    1. Token Embedding
    2. Positional Embedding
    3. Dropout
    4. Transformer Blocks
        - 12 (=num_layers)* TransformerBlock
        - Each TransformerBlock contains MHA (Multi-Head Attention),
        Feed-forward layer and Layer Norm    
    5. Final Norm (Layer Norm)
    6. Output Head
    """
    def __init__(self, config:Dict[str, int]) -> None:
        """
        Constructor to define the model architecture
        
        Args:
            config: configuration to construct the model architecture.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.positional_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config["num_layers"])
        ])
        self.final_norm = nn.LayerNorm(config["emb_dim"])
        self.output_head = nn.Linear(config["emb_dim"], config["vocab_size"])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_tokens]
            Input the tokenized sentence.
        """
        batch_size, num_tokens = x.shape
        # 1. Get the token embeddings
        # [batch_size, num_tokens] -> [batch_size, num_tokens, emb_dim]
        x = self.token_embedding(x)
        # 2. Get the positional embeddings
        # positional_embedding: [context_length, emb_dim]
        # x: [batch_size, num_tokens]
        # [batch_size, num_tokens] -> [batch_size, num_tokens, emb_dim]
        x = x + self.positional_embedding(torch.arange(num_tokens))
        # 3. Apply Dropout
        # [batch_size, num_tokens, emb_dim] -> [batch_size, num_tokens, emb_dim]
        x = self.dropout(x)
        # 4. Apply the Transformer Blocks
        # [batch_size, num_tokens, emb_dim] -> [batch_size, num_tokens, emb_dim]
        for block in self.transformer_blocks:
            x = block(x)
        # 5. Apply the final norm
        # [batch_size, num_tokens, emb_dim] -> [batch_size, num_tokens, emb_dim]
        x = self.final_norm(x)
        # 6. Apply the output head
        # [batch_size, num_tokens, emb_dim] -> [batch_size, num_tokens, vocab_size]
        x = self.output_head(x)
        return x



class TransformerBlock(nn.Module):
    """
    Transformer Block which contains Transformer layer (Multi-Head Attention),
    Feed forward layer, and Layer Norm.
    """
    def __init__(self, config: Dict[str, int]) -> None:
        """
        Constructor to define the model architecture
        
        Args:
            config: configuration to construct the model architecture.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            dim_in=config["emb_dim"],
            dim_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["num_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = nn.LayerNorm(config["emb_dim"])
        self.layer_norm_2 = nn.LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_tokens, emb_dim]
            Input the embedding for each batch and each token.
            emb_dim indicates the dim of embeddings.(emb_dim)
        """
        x2 = self.layer_norm_1(x)
        x2 = self.multi_head_attention(x2)
        x = x + self.dropout(x2)
        x2 = self.layer_norm_2(x)
        x2 = self.feed_forward(x2)
        x = x + self.dropout(x2)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention that enhances the interactions among the tokens,
    exploiting the self-attention mechanism to enhance the features relationship
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout: float, 
        num_heads: int,
        qkv_bias: bool,
    ) -> None:
        """        
        Args:
            dim_in: the dimension of embedding token
            dim_out: the dimension of output
            context_length: maximum length of the number of tokens as inputs
            dropout: dropout rate after the computation of Attention
            num_heads: the number of the multi-head attention
            qkv_bias: True to use bias to compute query/key/value in the attention
        """
        super().__init__()
        assert (dim_out % num_heads == 0), f"The dim of output should be divisible by num_heads"
        
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.dim_head = dim_out // num_heads

        # Q, K, V
        self.W_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        # Create a mask to prevent the attention from attending to future tokens
        self.mask = torch.triu(torch.ones(context_length, context_length))
        ## Set the mask to the lower triangular part of the matrix
        self.mask = self.mask.masked_fill(self.mask == 1, float("-inf"))

        # Final Linear layer
        self.W_o = nn.Linear(dim_out, dim_out)

        # Initialize the weights and biases
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset the weights and biases
        """
        self.W_q.reset_parameters()
        self.W_k.reset_parameters()
        self.W_v.reset_parameters()
        self.W_o.reset_parameters()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            [batch_size, num_tokens, dim_in]
            Input the embedding for each batch and each token.
            dim_in indicates the dim of embeddings.(emb_dim)

        Returns:
            [batch_size, num_tokens, dim_out]
            return the tensor with dim_out.
        """
        # Get the input shape
        batch_size, num_tokens, _ = x.shape
        # Generate Key, Query, and Value
        # [batch_size, num_tokens, dim_in]-> [batch_size, num_tokens, dim_out]
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # reshape the keys, queries, and values
        # [batch_size, num_tokens, dim_out]->[batch_size, num_tokens, num_heads, dim_head]
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.dim_head)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.dim_head)
        values = values.view(batch_size, num_tokens, self.num_heads, self.dim_head)

        # permute to swap the axis
        # [batch_size, num_heads, num_tokens, dim_head]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 1. Compute Attention Score
        # [B, n_h, N, d_h] * [B, n_h, d_h, N]
        attn_scores = queries @ keys.transpose(2, 3)

        # 2. Apply Dropout to the attention scores
        attn_scores = self.dropout(attn_scores)

        # 3. Apply the mask to the attention scores
        # [B, n_h, N, N]
        masked_attn_scores = attn_scores.masked_fill(self.mask[:num_tokens, :num_tokens], float("-inf"))

        # 4. Normalize the attention scores
        # [B, n_h, N, N]
        attn_scores_softmax = F.softmax(masked_attn_scores / np.sqrt(self.dim_head), dim=-1)

        # 5. Compute the weighted sum of the values
        # [B, n_h, N, N] * [B, n_h, N, d_h] = [B, n_h, N, d_h]
        out = attn_scores_softmax @ values

        # 6. Reshape and concatenate the outputs of the heads
        # [B, n_h, N, d_h]->[B, N, n_h * d_h] -> [B, N, dim_out]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.dim_out)

        # 7. Apply Dropout to the output
        # [B, N, dim_out] -> [B, N, dim_out]
        out = self.W_o(out)
        return out
        
        
class FeedForward(nn.Module):
    """
    Feed Forward layer that enhances the interactions among the tokens,
    exploiting the self-attention mechanism to enhance the features relationship
    """
    def __init__(self, config: Dict[str, int]) -> None:
        """
        Constructor to define the model architecture
        
        Args:
            config: configuration to construct the model architecture.
        """
        super().__init__()
        self.linear_1 = nn.Linear(config["emb_dim"], config["emb_dim"] * 4)
        self.linear_2 = nn.Linear(config["emb_dim"] * 4, config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_tokens, emb_dim]
            Input the embedding for each batch and each token.
            emb_dim indicates the dim of embeddings.(emb_dim)
        """
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        return x
        
        