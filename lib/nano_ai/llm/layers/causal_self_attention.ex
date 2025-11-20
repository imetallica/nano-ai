defmodule NanoAi.LLM.Layers.CausalSelfAttention do
  @moduledoc """
  Causal Self-Attention mechanism for autoregressive language models.

  ## Overview

  Self-attention allows each token in a sequence to gather information from other tokens.
  **Causal** means each token can only attend to itself and previous tokens, never future ones.
  This is essential for autoregressive generation (predicting the next token).

  ## The Core Computation

      Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

  Where:
  - **Q (Query)**: "What am I looking for?"
  - **K (Key)**: "What do I contain?"
  - **V (Value)**: "What information do I provide?"

  ## Step-by-Step Process

  ### 1. Linear Projections

  The input is projected into three different representations:

      input: [batch, seq_len, n_embd]

      Q = input × W_q    # [batch, seq_len, n_embd]
      K = input × W_k    # [batch, seq_len, n_embd]
      V = input × W_v    # [batch, seq_len, n_embd]

  Each token becomes:
  - A **query** (asking what to attend to)
  - A **key** (advertising its content)
  - A **value** (providing its information)

  ### 2. Multi-Head Split

  We split the embedding dimension into multiple heads for parallel attention:

      n_head = 6, n_embd = 768
      head_dim = n_embd / n_head = 128

      Q: [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_dim]
      K: [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_dim]
      V: [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_dim]

  Each head can learn different attention patterns:
  - Head 1: Syntax and grammar
  - Head 2: Semantic relationships
  - Head 3: Positional dependencies
  - Head 4: Long-range context
  - Head 5: Local word pairs
  - Head 6: Coreference resolution

  ### 3. Attention Scores

  Compute similarity between all query-key pairs:

      scores = Q × K^T / √head_dim    # [batch, n_head, seq_len, seq_len]

      # Example for 4 tokens (before scaling and masking):
      #            tok0  tok1  tok2  tok3
      # tok0   [   0.5   0.2   0.1   0.3  ]
      # tok1   [   0.4   0.6   0.2   0.1  ]
      # tok2   [   0.3   0.5   0.7   0.2  ]
      # tok3   [   0.2   0.3   0.4   0.8  ]

  `scores[i][j]` represents how much token i should attend to token j.

  ### 4. Scaling

      scores = scores / √head_dim

  **Why scale?**
  Prevents dot products from becoming too large, which would cause softmax
  to produce extreme (nearly one-hot) distributions. Maintains stable gradients.

  ### 5. Causal Masking (Critical!)

  Apply a lower triangular mask to prevent attending to future tokens:

      mask = [[1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 1]]

      scores = where(mask == 0, -∞, scores)

      # Result:
      #            tok0  tok1  tok2  tok3
      # tok0   [   0.5   -∞    -∞    -∞   ]  ← sees only itself
      # tok1   [   0.4   0.6   -∞    -∞   ]  ← sees tok0, itself
      # tok2   [   0.3   0.5   0.7   -∞   ]  ← sees tok0, tok1, itself
      # tok3   [   0.2   0.3   0.4   0.8  ]  ← sees all previous

  This ensures autoregressive property: token at position i only uses information
  from positions 0 to i.

  ### 6. Softmax

  Convert scores to attention weights (probabilities that sum to 1):

      attention_weights = softmax(scores, axis: -1)

      # -∞ becomes 0 after softmax:
      #            tok0  tok1  tok2  tok3
      # tok0   [   1.0   0.0   0.0   0.0  ]
      # tok1   [   0.4   0.6   0.0   0.0  ]
      # tok2   [   0.2   0.3   0.5   0.0  ]
      # tok3   [   0.1   0.2   0.3   0.4  ]

  Each row sums to 1.0 and represents how much each position attends to others.

  ### 7. Apply to Values

  Use attention weights to aggregate information from values:

      output = attention_weights × V    # [batch, n_head, seq_len, head_dim]

  Each output token is a weighted combination of all (previous) value vectors.

  ### 8. Merge Heads

  Concatenate all heads back together:

      [batch, n_head, seq_len, head_dim] → [batch, seq_len, n_embd]

  ### 9. Output Projection

  Final linear projection:

      output = merged × W_o    # [batch, seq_len, n_embd]

  ## Multi-Head Attention Benefits

  Instead of one attention operation, we run N parallel operations (heads):

  **Why multiple heads?**
  - Different heads can focus on different aspects:
    - Syntactic patterns (subject-verb agreement)
    - Semantic relationships (entity connections)
    - Positional information (nearby words)
    - Long-range dependencies (discourse structure)

  - Provides redundancy and robustness
  - Allows learning diverse attention patterns
  - More parameters without significantly more compute

  ## Causal Masking Deep Dive

  **Why is causal masking necessary?**

  During training:
  - We process entire sequences in parallel
  - Without masking, token at position i could "cheat" by looking at token i+1
  - The model would learn to just copy the next token, not truly predict it

  During generation:
  - We generate one token at a time
  - By masking during training, the model learns to work with partial sequences
  - This matches the generation scenario where future tokens don't exist yet

  **Example: Predicting "cat"**

      Sequence: "The cat sat"

      When predicting "cat":
      - Can attend to: "The" (position 0)
      - Can attend to: "cat" (position 1, itself)
      - Cannot attend to: "sat" (position 2, future)

      This forces the model to predict "sat" using only ["The", "cat"]

  ## Parameters

  For n_embd = 768, n_head = 6:

  - Q projection: 768 × 768 = 589,824
  - K projection: 768 × 768 = 589,824
  - V projection: 768 × 768 = 589,824
  - Output projection: 768 × 768 = 589,824
  - **Total: ~2.4M parameters per attention layer**

  ## Computational Complexity

  **Time complexity: O(n² × d)**
  - n = sequence length
  - d = embedding dimension
  - The bottleneck is the n² attention matrix

  For seq_len = 1024, n_embd = 768:
  - Attention matrix: 1024 × 1024 = 1M entries per head
  - With 6 heads: 6M attention scores
  - This is why long contexts are expensive!

  **Space complexity: O(n² × h)**
  - h = number of heads
  - Must store attention weights for backpropagation
  - Memory grows quadratically with sequence length

  ## Attention Pattern Examples

  Real attention patterns observed in trained models:

  **Syntactic Head**: Subject-verb agreement
  ```
  "The cats [attend heavily to 'The'] are sleeping"
  ```

  **Positional Head**: Attends to previous token
  ```
  Each token attends strongly to the immediately preceding token
  ```

  **Long-range Head**: Discourse coherence
  ```
  "John went to the store. ... He [attends to 'John'] bought milk."
  ```

  **Semantic Head**: Entity relationships
  ```
  "Paris [attends to] is the capital of France [attends to]"
  ```

  ## Usage

      # Single attention layer
      input = Axon.input("tokens", shape: {nil, nil, 768})
      output = CausalSelfAttention.layer(input, 768, 6, "attn")
      # Output shape: [batch, seq_len, 768]

      # In a transformer block
      block = input
        |> Axon.layer_norm()
        |> CausalSelfAttention.layer(768, 6, "block-0.attn")
        |> then(fn x -> Axon.add(input, x) end)  # Residual

  ## Performance Considerations

  **Memory optimizations:**
  - Cache the causal mask (currently recreated each forward pass)
  - Use flash attention for long sequences (when available)
  - Consider sparse attention patterns for very long contexts

  **Compute optimizations:**
  - Fused attention kernels (combine multiple ops)
  - Mixed precision (FP16 for attention scores)
  - KV caching during generation (reuse previous key/value pairs)

  ## Variants

  This module implements standard causal self-attention. Other variants:

  - **Multi-Query Attention (MQA)**: Share K, V across heads (faster inference)
  - **Grouped-Query Attention (GQA)**: Groups of heads share K, V (LLaMA 2)
  - **Flash Attention**: Memory-efficient implementation (2-4× faster)
  - **Sparse Attention**: Only attend to subset of tokens (Longformer, BigBird)
  - **Sliding Window**: Only attend to nearby tokens (local attention)

  ## References

  - Original attention: "Attention is All You Need" (Vaswani et al., 2017)
  - GPT-2: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
  - Causal masking: Essential for autoregressive models
  - Multi-head benefits: "Analyzing Multi-Head Self-Attention" (Voita et al., 2019)
  - Flash Attention: "Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
  """

  import Nx.Defn

  @doc """
  Builds a complete causal self-attention layer.

  This is the main entry point that orchestrates all attention operations:
  1. Project input to Q, K, V
  2. Split into multiple heads
  3. Compute attention scores
  4. Apply causal mask
  5. Apply softmax
  6. Apply attention to values
  7. Merge heads
  8. Final output projection

  ## Parameters

  - `input` - Input tensor [batch, seq_len, num_embed]
  - `num_embed` - Embedding dimension (e.g., 768)
  - `num_heads` - Number of attention heads (e.g., 6)
  - `name` - Layer name prefix for all sub-layers

  ## Returns

  Output tensor with same shape as input: [batch, seq_len, num_embed]

  ## Constraints

  - `num_embed` must be divisible by `num_heads`
  - Resulting `head_dim = num_embed / num_heads`
  - Each head operates on `head_dim` dimensions independently

  ## Example

      # Create attention layer
      input = Axon.input("x", shape: {nil, nil, 768})
      output = CausalSelfAttention.layer(input, 768, 6, "attn-0")

      # With 768 dimensions and 6 heads:
      # - Each head sees 128 dimensions (768 / 6)
      # - 6 parallel attention operations
      # - Results concatenated back to 768 dimensions

  ## Shape Flow

      Input:  [8, 1024, 768]         # batch=8, seq=1024, embed=768
        ↓
      Q,K,V:  [8, 1024, 768] each
        ↓
      Split:  [8, 6, 1024, 128]      # 6 heads, 128 dims each
        ↓
      Scores: [8, 6, 1024, 1024]     # attention matrix per head
        ↓
      Masked: [8, 6, 1024, 1024]     # with causal mask applied
        ↓
      Weights:[8, 6, 1024, 1024]     # after softmax
        ↓
      Applied:[8, 6, 1024, 128]      # attended values
        ↓
      Merged: [8, 1024, 768]         # concatenate heads
        ↓
      Output: [8, 1024, 768]         # final projection
  """
  def layer(input, num_embed, num_heads, name) do
    head_dim = div(num_embed, num_heads)

    {q, k, v} = project_qkv(input, num_embed, name)
    q = split_heads(q, num_heads, head_dim, name)
    k = split_heads(k, num_heads, head_dim, name)
    v = split_heads(v, num_heads, head_dim, name)
    scores = compute_attention_scores(q, k, head_dim, name)
    masked_scores = apply_causal_mask(scores, name)

    attention_weights = apply_softmax(masked_scores, name)
    attention_output = apply_attention_weights(attention_weights, v, name)
    merged_output = merge_heads(attention_output, name)
    final_projection(merged_output, num_embed, name)
  end

  # Step 1: Linear projections for Q, K, V
  defp project_qkv(input, num_embed, name) do
    # Implement linear projections for Q, K, V
    q = Axon.dense(input, num_embed, use_bias: false, name: "#{name}.q-proj")
    k = Axon.dense(input, num_embed, use_bias: false, name: "#{name}.k-proj")
    v = Axon.dense(input, num_embed, use_bias: false, name: "#{name}.v-proj")

    {q, k, v}
  end

  # Step 2: Split heads
  defp split_heads(x, num_heads, head_dim, name) do
    Axon.layer(
      fn x, _opts ->
        split_heads_fn(x, num_heads, head_dim)
      end,
      [x],
      name: "#{name}.split-heads",
      op_name: :split_heads
    )
  end

  defnp split_heads_fn(x, num_heads, head_dim) do
    {batch_size, seq_len, _} = Nx.shape(x)

    x
    |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Step 3: Compute attention scores
  defp compute_attention_scores(q, k, head_dim, name) do
    Axon.layer(
      fn q, k, _opts ->
        compute_attention_scores_fn(q, k, head_dim)
      end,
      [q, k],
      name: "#{name}.attention-scores",
      op_name: :attention_scores
    )
  end

  defnp compute_attention_scores_fn(q_tensor, k_tensor, head_dim) do
    # q: [batch, n_head, seq_len, head_dim]
    # k: [batch, n_head, seq_len, head_dim]

    # Transpose K: [batch, n_head, head_dim, seq_len]
    k_t = Nx.transpose(k_tensor, axes: [0, 1, 3, 2])

    # Q × K^T: [batch, n_head, seq_len, seq_len]
    scores = Nx.dot(q_tensor, [3], [0, 1], k_t, [2], [0, 1])
    scale = Nx.sqrt(head_dim)
    Nx.divide(scores, scale)
  end

  # Step 4: Apply causal mask
  # TODO OPTIMIZATION: Cache the mask to avoid recreation on every batch
  defp apply_causal_mask(scores, name) do
    Axon.layer(
      &apply_causal_mask_fn/2,
      [scores],
      name: "#{name}.causal-mask",
      op_name: :causal_mask
    )
  end

  defnp apply_causal_mask_fn(scores, _opts) do
    {batch_size, n_head, seq_len, _} = Nx.shape(scores)

    # Create lower triangular mask
    # 1 keep, 0 mask
    mask =
      {seq_len, seq_len}
      |> Nx.iota(axis: 0, type: :u8)
      |> Nx.greater_equal(Nx.iota({seq_len, seq_len}, axis: 1))
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch_size, n_head, seq_len, seq_len})

    neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))

    Nx.select(mask, scores, neg_inf)
  end

  # Step 5: Apply softmax to get attention weights
  defp apply_softmax(masked_scores, name) do
    Axon.softmax(masked_scores, axis: -1, name: "#{name}.softmax", op_name: :softmax)
  end

  # Step 6: Apply attention weights to values
  defp apply_attention_weights(attention_weights, v, name) do
    Axon.layer(
      &apply_attention_weights_fn/3,
      [attention_weights, v],
      name: "#{name}.apply-attention",
      op_name: :apply_attention
    )
  end

  defnp apply_attention_weights_fn(attention_weights, v, _opts) do
    # attention_weights:  [batch, n_head, seq_len, seq_len]
    # v_tensor: [batch, n_head, seq_len, head_dim]
    # output:   [batch, n_head, seq_len, head_dim]

    Nx.dot(attention_weights, [3], [0, 1], v, [2], [0, 1])
  end

  # Step 7: Combine heads and project output
  defp merge_heads(attention_output, name) do
    Axon.layer(
      &merge_heads_fn/2,
      [attention_output],
      name: "#{name}.merge-heads",
      op_name: :merge_heads
    )
  end

  defnp merge_heads_fn(heads, _opts) do
    # input: [batch, n_head, seq_len, head_dim]
    # output: [batch, seq_len, n_embd]

    {batch_size, n_head, seq_len, head_dim} = Nx.shape(heads)

    heads
    # [batch, seq_len, n_head, head_dim]
    |> Nx.transpose(axes: [0, 2, 1, 3])
    # [batch, seq_len, n_embd]
    |> Nx.reshape({batch_size, seq_len, n_head * head_dim})
  end

  defp final_projection(merged_output, num_embed, name) do
    Axon.dense(merged_output, num_embed, use_bias: false, name: "#{name}.out-proj")
  end
end
