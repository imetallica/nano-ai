defmodule NanoAi.LLM.Layers.FeedForward do
  @moduledoc """
  Feed-Forward Network (FFN) layer for transformer blocks.

  ## Overview

  The FFN is a simple two-layer neural network applied independently to each token position.
  It provides the transformer with non-linear transformation capacity and is where much of
  the model's "knowledge" is believed to be stored.

  ## Basic Architecture

      Input → Linear (expand) → Activation → Linear (contract) → Output

      [batch, seq_len, n_embd] → [batch, seq_len, 4×n_embd] → [batch, seq_len, n_embd]

  ## The Two Projections

  ### 1. Up Projection (Expansion)

      hidden = input × W_up    # [batch, seq_len, 768] → [batch, seq_len, 3072]

  Expands the representation to a higher dimension (typically 4× the embedding size).
  This gives the network more capacity to learn complex transformations.

  ### 2. Down Projection (Contraction)

      output = hidden × W_down    # [batch, seq_len, 3072] → [batch, seq_len, 768]

  Projects back to the original dimension after applying non-linearity.

  ## Why 4× Expansion?

  The 4× factor is empirically chosen across most transformers:
  - Large enough to provide significant representational capacity
  - Small enough to be computationally tractable
  - Consistent across GPT, BERT, LLaMA, and most modern transformers

  Some models use different ratios:
  - LLaMA with SwiGLU: 8/3× (≈2.67×) due to extra gate parameters
  - Dense models: 4×
  - Sparse models: Can use larger ratios (6-8×)

  ## Position-Wise Application

  The FFN is applied **independently** to each token position:

      Token 0: FFN(embedding[0]) → output[0]
      Token 1: FFN(embedding[1]) → output[1]
      Token 2: FFN(embedding[2]) → output[2]
      ...

  No information flows between positions in the FFN - that's the attention layer's job.
  This is why it's called "position-wise" feed-forward.

  ## Role in the Transformer

  The transformer block alternates between two operations:
  1. **Attention**: Mixes information ACROSS positions (communication)
  2. **FFN**: Transforms EACH position independently (computation)

  Think of it as:
  - Attention = "gather relevant information from other tokens"
  - FFN = "process and transform that information"

  ## Parameters

  For n_embd = 768 and expand_factor = 4:
  - W_up: [768, 3072] = 2,359,296 parameters
  - W_down: [3072, 768] = 2,359,296 parameters
  - **Total: ~4.7M parameters per FFN layer**

  In a 12-layer model, FFN accounts for roughly 2/3 of all parameters!

  ## Activation Functions

  Different activation functions provide different non-linear transformations:

  ### GELU (Gaussian Error Linear Unit)
  Used in GPT-2, GPT-3, BERT.

      GELU(x) = x × Φ(x)

  Where Φ(x) is the cumulative distribution function of the standard normal distribution.

  **Properties:**
  - Smooth approximation of ReLU
  - Probabilistic interpretation (dropout-like)
  - Better gradient flow than ReLU
  - Standard choice for most transformers

  ### SiLU / Swish (Sigmoid Linear Unit)
  Used in LLaMA, PaLM.

      SiLU(x) = x × sigmoid(x)

  **Properties:**
  - Smooth, non-monotonic
  - Better than ReLU for deeper networks
  - Used in modern efficient architectures

  ### ReLU (Rectified Linear Unit)
  Used in original Transformer.

      ReLU(x) = max(0, x)

  **Properties:**
  - Simple and fast
  - Can have "dead neurons" (always output 0)
  - Less common in modern LLMs

  ## FFN Variants

  ### Standard FFN (GPT-2, BERT)

      hidden = activation(input × W_up)
      output = hidden × W_down

  Simple two-layer network with one activation function.

  ### Gated Linear Unit (GLU) Variants (LLaMA, PaLM)

      gate = activation(input × W_gate)
      hidden = input × W_up
      output = (gate ⊙ hidden) × W_down

  Adds an extra "gate" that controls information flow.

  **Types:**
  - **ReGLU**: ReLU-gated (gate = ReLU(input × W_gate))
  - **GEGLU**: GELU-gated (gate = GELU(input × W_gate))
  - **SiGLU/SwiGLU**: SiLU-gated (gate = SiLU(input × W_gate))

  **Benefits:**
  - Often improves performance (5-10% better perplexity)
  - More parameters but better parameter efficiency
  - Standard in LLaMA, PaLM, and modern efficient models

  **Drawbacks:**
  - Extra W_gate parameters (~33% more parameters)
  - Slightly more computation

  ## Parameter Comparison

  For n_embd = 768, expand_factor = 4:

  | Variant | Up/Gate | Down | Total | Notes |
  |---------|---------|------|-------|-------|
  | Standard GELU | 2.4M | 2.4M | 4.7M | GPT-2/3 |
  | ReGLU/GEGLU/SwiGLU | 2.4M + 2.4M | 2.4M | 7.1M | +50% params |

  Gated variants use more parameters but often achieve better quality per parameter.

  ## Memory and Compute

  For batch_size=8, seq_len=1024, n_embd=768, expand_factor=4:

  **Activations:**
  - Input: 8 × 1024 × 768 = 6.3M floats = 25 MB
  - Hidden: 8 × 1024 × 3072 = 25.2M floats = 100 MB
  - Output: 8 × 1024 × 768 = 6.3M floats = 25 MB

  **FLOPs per token:**
  - Up projection: 768 × 3072 = 2.4M FLOPs
  - Down projection: 3072 × 768 = 2.4M FLOPs
  - **Total: ~4.8M FLOPs per token**

  For 1024 tokens: ~4.9 GFLOPs per sequence

  ## Usage

      # Standard GELU FFN (GPT-2/3 style)
      output = FeedForward.gelu(input, 768, "ffn", expand_factor: 4)

      # SwiGLU (LLaMA style)
      output = FeedForward.siglu(input, 768, "ffn", expand_factor: 4)

      # GEGLU (T5 style)
      output = FeedForward.geglu(input, 768, "ffn", expand_factor: 4)

      # ReGLU (ReLU-gated)
      output = FeedForward.reglu(input, 768, "ffn", expand_factor: 4)

  ## Which Variant to Choose?

  **For most use cases: Standard GELU**
  - Proven, stable, widely used
  - Good balance of performance and simplicity
  - Used in GPT-2, GPT-3, BERT

  **For efficiency: SwiGLU**
  - Better performance per parameter
  - Used in LLaMA (state-of-the-art efficient models)
  - +50% parameters but >10% better quality

  **For research: Experiment with all**
  - Different tasks may benefit from different activations
  - Gated variants generally perform better but cost more

  ## Design Decisions

  - **No bias terms**: Common practice in modern LLMs (GPT-3, LLaMA)
    - Saves parameters without hurting performance
    - Layer norm before FFN makes bias redundant

  - **Separate up/gate projections**: For gated variants
    - Allows independent learning of gate and value
    - More flexible than single projection

  - **4× expansion**: Standard across most transformers
    - Empirically proven to work well
    - Good capacity/compute trade-off

  ## Performance Tips

  1. **Use GELU for initial experiments** - Stable and well-understood
  2. **Try SwiGLU for production models** - Better quality if you have compute budget
  3. **Increase expand_factor if overfitting** - More capacity can help
  4. **Decrease expand_factor if compute-limited** - 2-3× can work for smaller models

  ## References

  - Original FFN: "Attention is All You Need" (Vaswani et al., 2017)
  - GELU: "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
  - GLU variants: "GLU Variants Improve Transformer" (Shazeer, 2020)
  - SwiGLU in LLaMA: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
  """

  @doc """
  Standard FFN with GELU activation (GPT-2/3, BERT style).

  This is the most common FFN variant, used in GPT-2, GPT-3, and BERT.

  ## Architecture

      Input: [batch, seq_len, n_embed]
        ↓
      Up Projection: [batch, seq_len, n_embed × expand_factor]
        ↓
      GELU Activation
        ↓
      Down Projection: [batch, seq_len, n_embed]

  ## Formula

      hidden = input × W_up              # Expand
      activated = GELU(hidden)           # Non-linearity
      output = activated × W_down        # Contract

  Where GELU(x) = x × Φ(x), and Φ is the CDF of the standard normal distribution.

  ## Parameters

  - `input` - Input tensor [batch, seq_len, n_embed]
  - `n_embed` - Embedding dimension (e.g., 768)
  - `name` - Layer name prefix
  - `opts` - Options:
    - `:expand_factor` - Hidden size multiplier. Default: 4 (768 → 3072 → 768)

  ## Returns

  Output tensor with same shape as input: [batch, seq_len, n_embed]

  ## Example

      input = Axon.input("x", shape: {nil, nil, 768})
      output = FeedForward.gelu(input, 768, "ffn", expand_factor: 4)
      # Output shape: [batch, seq_len, 768]

  ## When to Use

  - **Default choice** for most applications
  - Proven stable and effective
  - Use this unless you have specific requirements for gated variants
  """
  @spec gelu(
          input :: Axon.t(),
          n_embed :: integer(),
          opts :: keyword()
        ) :: Axon.t()
  def gelu(input, n_embed, opts \\ [expand_factor: 4]) do
    hidden_size = n_embed * opts[:expand_factor]

    input
    |> up_projection(hidden_size)
    |> activation_layer(:gelu)
    |> down_projection(n_embed)
  end

  @doc """
  SwiGLU: Swish/SiLU-Gated Linear Unit (LLaMA style).

  A gated FFN variant used in LLaMA and other efficient modern models.
  Provides better performance than standard GELU at the cost of more parameters.

  ## Architecture

      Input: [batch, seq_len, n_embed]
        ↓
      Split into two paths:
        Gate Path: input × W_gate → SiLU → gate
        Value Path: input × W_up → value
        ↓
      Element-wise Multiply: gate ⊙ value
        ↓
      Down Projection: [batch, seq_len, n_embed]

  ## Formula

      gate = SiLU(input × W_gate)        # Gate with SiLU activation
      value = input × W_up               # Value without activation
      hidden = gate ⊙ value              # Element-wise multiplication
      output = hidden × W_down           # Contract

  Where SiLU(x) = x × sigmoid(x) = x / (1 + e^(-x))

  ## Parameters

  - `input` - Input tensor [batch, seq_len, n_embed]
  - `n_embed` - Embedding dimension (e.g., 768)
  - `name` - Layer name prefix
  - `opts` - Options:
    - `:expand_factor` - Hidden size multiplier. Default: 4

  ## Returns

  Output tensor with same shape as input: [batch, seq_len, n_embed]

  ## Parameter Count

  For n_embed=768, expand_factor=4:
  - Gate projection: 768 × 3072 = 2.4M
  - Up projection: 768 × 3072 = 2.4M
  - Down projection: 3072 × 768 = 2.4M
  - **Total: ~7.1M parameters** (50% more than standard GELU)

  ## Example

      input = Axon.input("x", shape: {nil, nil, 768})
      output = FeedForward.siglu(input, 768, "ffn", expand_factor: 4)

  ## When to Use

  - **For state-of-the-art performance**: Used in LLaMA and modern efficient models
  - When you have compute budget for extra parameters
  - Provides ~10% better quality for 50% more parameters
  - Industry standard for new efficient models

  ## Benefits vs Standard GELU

  - Better performance per parameter
  - More expressive gating mechanism
  - Proven in LLaMA (65B model matches 540B models on some tasks)
  """
  @spec siglu(
          input :: Axon.t(),
          n_embed :: integer(),
          opts :: keyword()
        ) :: Axon.t()
  def siglu(input, n_embed, opts \\ [expand_factor: 4]) do
    hidden_size = n_embed * opts[:expand_factor]

    input
    |> gate_projection(hidden_size)
    |> activation_layer(:silu)
    |> Axon.multiply(up_projection(input, hidden_size))
    |> down_projection(n_embed)
  end

  @doc """
  ReGLU: ReLU-Gated Linear Unit.

  A gated FFN variant using ReLU activation. Simpler than GEGLU/SwiGLU but still
  provides benefits of gating.

  ## Architecture

      Input: [batch, seq_len, n_embed]
        ↓
      Split into two paths:
        Gate Path: input × W_gate → ReLU → gate
        Value Path: input × W_up → value
        ↓
      Element-wise Multiply: gate ⊙ value
        ↓
      Down Projection: [batch, seq_len, n_embed]

  ## Formula

      gate = ReLU(input × W_gate)        # Gate with ReLU
      value = input × W_up               # Value without activation
      hidden = gate ⊙ value              # Element-wise multiplication
      output = hidden × W_down           # Contract

  Where ReLU(x) = max(0, x)

  ## Parameters

  - `input` - Input tensor [batch, seq_len, n_embed]
  - `n_embed` - Embedding dimension (e.g., 768)
  - `name` - Layer name prefix
  - `opts` - Options:
    - `:expand_factor` - Hidden size multiplier. Default: 4

  ## Returns

  Output tensor with same shape as input: [batch, seq_len, n_embed]

  ## Example

      input = Axon.input("x", shape: {nil, nil, 768})
      output = FeedForward.reglu(input, 768, "ffn", expand_factor: 4)

  ## When to Use

  - Simple alternative to GEGLU/SwiGLU
  - Faster computation than smooth activations
  - Good for experimentation
  - Less common in production models (GELU/SwiGLU preferred)

  ## Trade-offs

  - **Pros**: Simple, fast, gated benefits
  - **Cons**: "Dead ReLU" problem (neurons can stop learning)
  - GELU and SwiGLU typically perform better
  """
  @spec reglu(
          input :: Axon.t(),
          n_embed :: integer(),
          opts :: keyword()
        ) :: Axon.t()
  def reglu(input, n_embed, opts \\ [expand_factor: 4]) do
    hidden_size = n_embed * opts[:expand_factor]

    input
    |> gate_projection(hidden_size)
    |> activation_layer(:relu)
    |> Axon.multiply(up_projection(input, hidden_size))
    |> down_projection(n_embed)
  end

  @doc """
  GEGLU: GELU-Gated Linear Unit (T5 style).

  A gated FFN variant using GELU activation. Combines the benefits of GELU's
  smooth gradients with gating mechanisms.

  ## Architecture

      Input: [batch, seq_len, n_embed]
        ↓
      Split into two paths:
        Gate Path: input × W_gate → GELU → gate
        Value Path: input × W_up → value
        ↓
      Element-wise Multiply: gate ⊙ value
        ↓
      Down Projection: [batch, seq_len, n_embed]

  ## Formula

      gate = GELU(input × W_gate)        # Gate with GELU
      value = input × W_up               # Value without activation
      hidden = gate ⊙ value              # Element-wise multiplication
      output = hidden × W_down           # Contract

  Where GELU(x) = x × Φ(x)

  ## Parameters

  - `input` - Input tensor [batch, seq_len, n_embed]
  - `n_embed` - Embedding dimension (e.g., 768)
  - `name` - Layer name prefix
  - `opts` - Options:
    - `:expand_factor` - Hidden size multiplier. Default: 4

  ## Returns

  Output tensor with same shape as input: [batch, seq_len, n_embed]

  ## Parameter Count

  Same as other gated variants:
  - Gate projection: 2.4M
  - Up projection: 2.4M
  - Down projection: 2.4M
  - **Total: ~7.1M parameters**

  ## Example

      input = Axon.input("x", shape: {nil, nil, 768})
      output = FeedForward.geglu(input, 768, "ffn", expand_factor: 4)

  ## When to Use

  - Alternative to SwiGLU with GELU's properties
  - Used in some T5 variants
  - Good choice if you prefer GELU over SiLU
  - Smooth gradients like standard GELU

  ## Comparison

  - **vs Standard GELU**: Better performance, more parameters
  - **vs SwiGLU**: Similar performance, GELU properties instead of SiLU
  - **vs ReGLU**: Smoother, better gradient flow, slightly slower

  ## Performance

  Typically performs similarly to SwiGLU:
  - ~5-10% better than standard GELU
  - Comparable to other gated variants
  - Choice often comes down to activation preference
  """
  @spec geglu(
          input :: Axon.t(),
          n_embed :: integer(),
          opts :: keyword()
        ) :: Axon.t()
  def geglu(input, n_embed, opts \\ [expand_factor: 4]) do
    hidden_size = n_embed * opts[:expand_factor]

    input
    |> gate_projection(hidden_size)
    |> activation_layer(:gelu)
    |> Axon.multiply(up_projection(input, hidden_size))
    |> down_projection(n_embed)
  end

  defp gate_projection(input, hidden_size) do
    Axon.dense(input, hidden_size, use_bias: false)
  end

  defp up_projection(input, hidden_size) do
    Axon.dense(input, hidden_size, use_bias: false)
  end

  defp activation_layer(input, activation) do
    case activation do
      :gelu -> Axon.gelu(input)
      :silu -> Axon.silu(input)
      :relu -> Axon.relu(input)
    end
  end

  defp down_projection(input, num_embed) do
    Axon.dense(input, num_embed, use_bias: false)
  end
end
