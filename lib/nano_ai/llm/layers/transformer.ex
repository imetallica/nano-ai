defmodule NanoAi.LLM.Layers.Transformer do
  @moduledoc """
  A single Transformer Block combining attention and feed-forward layers.

  ## Overview

  The Transformer Block is the core building block of GPT-like models.
  It consists of two sub-layers, each with layer normalization and residual connections:

      1. Multi-Head Causal Self-Attention
      2. Position-wise Feed-Forward Network

  ## Architecture

  There are two common arrangements:

  ### Post-Norm (Original Transformer, GPT-2)

      x = LayerNorm(x + Attention(x))
      x = LayerNorm(x + FFN(x))

  Normalization happens AFTER the residual addition.

  ### Pre-Norm (GPT-3, LLaMA, most modern models)

      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

  Normalization happens BEFORE the sub-layer operation.

  Pre-Norm is more stable for training deep networks.
  Post-Norm can achieve slightly better final performance but requires careful learning rate warmup and is harder to train at scale.

  ## Residual Connections

  The "x + ..." pattern is crucial:

      output = input + SubLayer(input)

  Benefits:
  - **Gradient flow**: Gradients can flow directly through the addition
  - **Identity mapping**: Layer can learn to do nothing (pass input through)
  - **Easier optimization**: Makes training very deep networks feasible

  Without residuals, training 12+ layer networks would be extremely difficult.

  ## Layer Normalization

  Normalizes activations to have zero mean and unit variance:

      LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) * γ + β

  Where γ (scale) and β (shift) are learnable parameters.

  Benefits:
  - Stabilizes training
  - Reduces internal covariate shift
  - Allows higher learning rates

  ## Data Flow Through One Block

      Input: [batch, seq_len, num_embed]
        ↓
      LayerNorm
        ↓
      Causal Self-Attention (tokens communicate)
        ↓
      + (residual connection)
        ↓
      LayerNorm
        ↓
      Feed-Forward Network (tokens compute independently)
        ↓
      + (residual connection)
        ↓
      Output: [batch, seq_len, num_embed]

  Note: Input and output shapes are identical, allowing blocks to be stacked.

  ## Stacking Blocks

  GPT models stack multiple transformer blocks:

      Embeddings → Block 1 → Block 2 → ... → Block N → Output

  Each block refines the representations:
  - Early blocks: Learn low-level patterns (syntax, local context)
  - Middle blocks: Learn compositional features
  - Later blocks: Learn high-level semantics and task-specific features

  ## Parameters per Block

  For NanoAi config:
  - num_embed = 768
  - num_heads = 6
  - num_layers = 12

  Attention:
  - Q, K, V projections: 3 × (768 × 768) = 1,769,472
  - Output projection: 768 × 768 = 589,824
  - Subtotal: ~2.4M parameters

  FFN (4× expansion):
  - Up projection: 768 × 3072 = 2,359,296
  - Down projection: 3072 × 768 = 2,359,296
  - Subtotal: ~4.7M parameters

  Layer Norms:
  - 2 × (768 + 768) = 3,072 (scale and shift)

  Total per block: ~7.1M parameters

  For 12 blocks (num_layers = 12): ~85M parameters (just in transformer blocks!)

  ## Why This Structure Works

  1. **Attention**: Gathers relevant information from context
  2. **FFN**: Processes and transforms that information
  3. **Residuals**: Ensure stable gradient flow
  4. **LayerNorm**: Keeps activations well-behaved
  5. **Stacking**: Builds increasingly abstract representations
  """
  alias NanoAi.LLM.Layers.CausalSelfAttention
  alias NanoAi.LLM.Layers.FeedForward

  @doc """
  Creates a transformer block with pre-norm architecture (GPT-3, LLaMA style).

      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

  ## Parameters
    - input: Axon layer [batch, seq_len, num_embed]
    - num_embed: Embedding dimension
    - num_heads: Number of attention heads
    - name: Base name for the block
    - opts: Options
      - :ffn_type - FFN variant (:gelu, :reglu, etc.). Default: :gelu
      - :expand_factor - FFN expansion ratio. Default: 4
  """
  @spec pre_norm(
          input :: Axon.t(),
          num_embed :: integer(),
          num_heads :: integer(),
          name :: String.t(),
          opts :: keyword()
        ) :: Axon.t()
  def pre_norm(input, num_embed, num_heads, name, opts \\ []) do
    ffn_type = Keyword.get(opts, :ffn_type, :gelu)
    expand_factor = Keyword.get(opts, :expand_factor, 4)

    input
    |> pre_norm_attention_sublayer(num_embed, num_heads, name)
    |> pre_norm_ffn_sublayer(num_embed, name, ffn_type, expand_factor)
  end

  @doc """
  Creates a transformer block with post-norm architecture (Original Transformer, GPT-2 style).

      x = LayerNorm(x + Attention(x))
      x = LayerNorm(x + FFN(x))

  ## Parameters
    - input: Axon layer [batch, seq_len, num_embed]
    - num_embed: Embedding dimension
    - num_heads: Number of attention heads
    - name: Base name for the block
    - opts: Options
      - :ffn_type - FFN variant (:gelu, :reglu, etc.). Default: :gelu
      - :expand_factor - FFN expansion ratio. Default: 4
  """
  @spec post_norm(
          input :: Axon.t(),
          num_embed :: integer(),
          num_heads :: integer(),
          name :: String.t(),
          opts :: keyword()
        ) :: Axon.t()
  def post_norm(input, num_embed, num_heads, name, opts \\ []) do
    ffn_type = Keyword.get(opts, :ffn_type, :gelu)
    expand_factor = Keyword.get(opts, :expand_factor, 4)

    input
    |> post_norm_attention_sublayer(num_embed, num_heads, name)
    |> post_norm_ffn_sublayer(num_embed, name, ffn_type, expand_factor)
  end

  defp pre_norm_attention_sublayer(input, num_embed, num_heads, name) do
    input
    |> Axon.layer_norm(name: "#{name}.attn-norm")
    |> CausalSelfAttention.layer(
      num_embed,
      num_heads,
      "#{name}.attn"
    )
    |> then(fn x ->
      Axon.add(input, x, name: "#{name}.attn-residual")
    end)
  end

  defp pre_norm_ffn_sublayer(input, num_embed, name, ffn_type, expand_factor) do
    input
    |> Axon.layer_norm(name: "#{name}.ffn-norm")
    |> apply_ffn(num_embed, "#{name}.ffn", ffn_type, expand_factor)
    |> then(fn y ->
      Axon.add(input, y, name: "#{name}.ffn-residual")
    end)
  end

  defp post_norm_attention_sublayer(input, num_embed, num_heads, name) do
    input
    |> CausalSelfAttention.layer(
      num_embed,
      num_heads,
      "#{name}.attn"
    )
    |> then(fn attn_output ->
      input
      |> Axon.add(attn_output, name: "#{name}.attn-residual")
      |> Axon.layer_norm(name: "#{name}.attn-norm")
    end)
  end

  defp post_norm_ffn_sublayer(input, num_embed, name, ffn_type, expand_factor) do
    input
    |> apply_ffn(num_embed, "#{name}.ffn", ffn_type, expand_factor)
    |> then(fn ffn_output ->
      input
      |> Axon.add(ffn_output, name: "#{name}.ffn-residual")
      |> Axon.layer_norm(name: "#{name}.ffn-norm")
    end)
  end

  defp apply_ffn(input, n_embed, name, ffn_type, expand_factor) do
    case ffn_type do
      :gelu ->
        FeedForward.gelu(input, n_embed, name, expand_factor: expand_factor)

      :reglu ->
        FeedForward.reglu(input, n_embed, name, expand_factor: expand_factor)

      :geglu ->
        FeedForward.geglu(input, n_embed, name, expand_factor: expand_factor)

      :gated ->
        FeedForward.gated(input, n_embed, name, expand_factor: expand_factor)
    end
  end
end
