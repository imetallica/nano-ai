defmodule NanoAi.LLM.Layers.Transformer do
  @moduledoc """
  A single Transformer Block combining attention and feed-forward layers.

  ## Overview

  The Transformer Block is the fundamental building block of GPT-like models.
  It consists of two sub-layers, each with layer normalization and residual connections:

  1. **Multi-Head Causal Self-Attention**: Allows tokens to gather information from previous tokens
  2. **Position-wise Feed-Forward Network**: Processes each token independently

  Multiple blocks are stacked (typically 12-24) to create increasingly abstract representations.

  ## Architecture Styles

  ### Pre-Norm (GPT-3, LLaMA, Modern LLMs)

      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

  Normalization happens BEFORE the sub-layer operation.

  **Benefits:**
  - More stable training for deep networks (12+ layers)
  - Gradients flow more easily
  - Can use higher learning rates
  - Standard in modern LLMs

  ### Post-Norm (Original Transformer, GPT-2)

      x = LayerNorm(x + Attention(x))
      x = LayerNorm(x + FFN(x))

  Normalization happens AFTER the residual addition.

  **Benefits:**
  - Can achieve slightly better final performance
  - Closer to original "Attention is All You Need" paper

  **Drawbacks:**
  - Harder to train (requires careful learning rate warmup)
  - Less stable for very deep networks (24+ layers)
  - Gradients can explode/vanish more easily

  ## Data Flow Through One Block

      Input: [batch, seq_len, num_embed]
        ↓
      LayerNorm (pre-norm) or skip
        ↓
      Causal Self-Attention
      • Tokens gather context from previous tokens
      • Multi-head allows learning different patterns
      • Output: [batch, seq_len, num_embed]
        ↓
      + Residual Connection (add original input)
        ↓
      LayerNorm (post-norm) or for next sub-layer
        ↓
      Feed-Forward Network
      • Expand: [batch, seq_len, num_embed × 4]
      • Apply non-linearity (GELU, ReLU, etc.)
      • Contract: [batch, seq_len, num_embed]
        ↓
      + Residual Connection
        ↓
      Output: [batch, seq_len, num_embed]

  Note: Input and output shapes are identical, allowing blocks to be stacked.

  ## Residual Connections

  The "x + ..." pattern is crucial for training deep networks:

      output = input + SubLayer(input)

  **Why residuals matter:**
  - **Gradient flow**: Gradients can flow directly through the addition operation
  - **Identity mapping**: Each layer can learn to do nothing (just pass input through)
  - **Easier optimization**: Without residuals, training 12+ layer networks would be nearly impossible
  - **Mitigates vanishing gradients**: Each layer adds to the signal rather than transforming it entirely

  ## Layer Normalization

  Normalizes activations to have zero mean and unit variance:

      LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) * γ + β

  Where γ (scale) and β (shift) are learned parameters.

  **Benefits:**
  - Stabilizes training by keeping activations in a reasonable range
  - Reduces internal covariate shift (how much layer inputs change during training)
  - Allows higher learning rates
  - Makes the model less sensitive to initialization

  ## Parameters per Block

  For default configuration (num_embed=768, num_heads=6, expand_factor=4):

  **Attention Layer:**
  - Q projection: 768 × 768 = 589,824
  - K projection: 768 × 768 = 589,824
  - V projection: 768 × 768 = 589,824
  - Output projection: 768 × 768 = 589,824
  - Subtotal: ~2.4M parameters

  **Feed-Forward Network (4× expansion):**
  - Up projection: 768 × 3,072 = 2,359,296
  - Down projection: 3,072 × 768 = 2,359,296
  - Subtotal: ~4.7M parameters

  **Layer Norms:**
  - 2 × (768 scale + 768 shift) = 3,072 parameters (negligible)

  **Total per block: ~7.1M parameters**

  For 12 blocks: ~85M parameters (nearly half the total model size!)

  ## Stacking Blocks

  GPT models stack multiple transformer blocks:

      Embeddings → Block 1 → Block 2 → ... → Block N → Output

  Each block refines the representations:
  - **Early blocks (1-4)**: Learn low-level patterns
    - Syntax, grammar, word order
    - Local dependencies (word pairs, simple phrases)

  - **Middle blocks (5-8)**: Learn compositional features
    - Phrase structure, argument relations
    - Medium-range dependencies (sentence structure)

  - **Later blocks (9-12)**: Learn high-level semantics
    - Abstract meaning, discourse structure
    - Long-range dependencies, reasoning

  ## FFN Variants

  The module supports multiple feed-forward network types:

  - `:gelu` - Standard GELU activation (GPT-2/3, BERT)
  - `:reglu` - ReLU-based gated linear unit
  - `:geglu` - GELU-based gated linear unit (used in some T5 variants)
  - `:gated` - Custom gated activation

  Gated variants typically perform better but use more parameters.

  ## Why This Structure Works

  1. **Attention**: Gathers relevant information from context
     - "The cat sat on the mat" - 'sat' can attend to 'cat' to know who's sitting

  2. **FFN**: Processes and transforms that information
     - Applies non-linear transformations to create richer representations
     - Stores factual knowledge and patterns

  3. **Residuals**: Ensure stable gradient flow
     - Allow the model to learn incremental refinements
     - Prevent gradient vanishing in deep networks

  4. **LayerNorm**: Keeps activations well-behaved
     - Prevents numerical instability
     - Makes training more robust to hyperparameter choices

  5. **Stacking**: Builds increasingly abstract representations
     - Each layer adds another level of understanding
     - Deeper networks can capture more complex patterns

  ## Usage

      # Pre-norm block (recommended for deep networks)
      block = Transformer.pre_norm(input, 768, 6, "block-0",
        ffn_type: :gelu,
        expand_factor: 4
      )

      # Post-norm block (original transformer style)
      block = Transformer.post_norm(input, 768, 6, "block-0",
        ffn_type: :gelu,
        expand_factor: 4
      )

      # Stack multiple blocks
      output = Enum.reduce(0..11, input, fn i, acc ->
        Transformer.pre_norm(acc, 768, 6, "block-\#{i}")
      end)

  ## Performance Considerations

  - **Pre-norm** is the recommended default for most use cases
  - Use **post-norm** only if you have specific requirements or want to match original papers
  - Larger **expand_factor** (6-8) can improve quality but increases parameters and compute
  - More **num_heads** allows learning more diverse attention patterns

  ## References

  - Original Transformer: "Attention is All You Need" (Vaswani et al., 2017)
  - Pre-norm benefits: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
  - GPT-3: Uses pre-norm architecture throughout
  - LLaMA: Modern pre-norm implementation with RMSNorm
  """
  alias NanoAi.LLM.Layers.CausalSelfAttention
  alias NanoAi.LLM.Layers.FeedForward

  @doc """
  Creates a transformer block with pre-norm architecture (GPT-3, LLaMA style).

  Pre-norm applies layer normalization BEFORE each sub-layer:

      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

  This is the modern standard for transformer models and provides better training stability
  for deep networks (12+ layers).

  ## Flow

      Input: [batch, seq_len, num_embed]
        ↓
      LayerNorm
        ↓
      Causal Self-Attention
        ↓
      + (add input - residual connection)
        ↓
      LayerNorm
        ↓
      Feed-Forward Network
        ↓
      + (add input - residual connection)
        ↓
      Output: [batch, seq_len, num_embed]

  ## Parameters

  - `input` - Axon layer with shape [batch, seq_len, num_embed]
  - `num_embed` - Embedding/hidden dimension (e.g., 768)
  - `num_heads` - Number of attention heads (e.g., 6)
  - `name` - Base name for this block (e.g., "block-0")
  - `opts` - Options:
    - `:ffn_type` - FFN activation (:gelu, :reglu, :geglu, :gated). Default: :gelu
    - `:expand_factor` - FFN expansion ratio. Default: 4 (768 → 3072 → 768)

  ## Returns

  Axon layer with the same shape as input: [batch, seq_len, num_embed]

  ## Example

      input = Axon.input("tokens", shape: {nil, nil, 768})

      # Single pre-norm block
      output = Transformer.pre_norm(input, 768, 6, "block-0",
        ffn_type: :gelu,
        expand_factor: 4
      )

      # Stack 12 pre-norm blocks
      model = Enum.reduce(0..11, input, fn i, acc ->
        Transformer.pre_norm(acc, 768, 6, "block-\#{i}")
      end)

  ## Why Pre-Norm?

  - **More stable training**: Gradients flow more smoothly through deep networks
  - **Higher learning rates**: Can train faster without diverging
  - **Better for 12+ layers**: Essential for very deep models (GPT-3 has 96 layers!)
  - **Industry standard**: Used in GPT-3, LLaMA, PaLM, and most modern LLMs
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
  Creates a transformer block with post-norm architecture (original Transformer, GPT-2 style).

  Post-norm applies layer normalization AFTER the residual addition:

      x = LayerNorm(x + Attention(x))
      x = LayerNorm(x + FFN(x))

  This is the original architecture from "Attention is All You Need" and is still used
  in some models (GPT-2, original BERT), but has largely been replaced by pre-norm.

  ## Flow

      Input: [batch, seq_len, num_embed]
        ↓
      Causal Self-Attention
        ↓
      + (add input - residual connection)
        ↓
      LayerNorm
        ↓
      Feed-Forward Network
        ↓
      + (add input - residual connection)
        ↓
      LayerNorm
        ↓
      Output: [batch, seq_len, num_embed]

  ## Parameters

  - `input` - Axon layer with shape [batch, seq_len, num_embed]
  - `num_embed` - Embedding/hidden dimension (e.g., 768)
  - `num_heads` - Number of attention heads (e.g., 6)
  - `name` - Base name for this block (e.g., "block-0")
  - `opts` - Options:
    - `:ffn_type` - FFN activation (:gelu, :reglu, :geglu, :gated). Default: :gelu
    - `:expand_factor` - FFN expansion ratio. Default: 4 (768 → 3072 → 768)

  ## Returns

  Axon layer with the same shape as input: [batch, seq_len, num_embed]

  ## Example

      input = Axon.input("tokens", shape: {nil, nil, 768})

      # Single post-norm block
      output = Transformer.post_norm(input, 768, 6, "block-0",
        ffn_type: :gelu,
        expand_factor: 4
      )

  ## When to Use Post-Norm?

  Use post-norm if:
  - You want to match the original Transformer paper exactly
  - You're reproducing GPT-2 or original BERT
  - You have a shallow model (6-8 layers) and want slightly better final performance

  ## Drawbacks

  - **Harder to train**: Requires careful learning rate warmup (1000-4000 steps)
  - **Less stable**: Gradients can explode or vanish more easily
  - **Not scalable**: Very difficult to train 24+ layer models
  - **Slower convergence**: Takes longer to reach good performance

  ## Recommendation

  **Use `pre_norm/5` instead** for most applications. Post-norm is mainly useful for
  research reproducibility or matching specific published architectures.
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

      :siglu ->
        FeedForward.siglu(input, n_embed, name, expand_factor: expand_factor)
    end
  end
end
