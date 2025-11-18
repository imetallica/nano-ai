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
  - Q (Query): "What am I looking for?"
  - K (Key): "What do I contain?"
  - V (Value): "What information do I provide?"

  ## Step-by-Step Process

  ### 1. Linear Projections

  The input is projected into three different representations:

      input: [batch, seq_len, n_embd]

      Q = input × W_q    # [batch, seq_len, n_embd]
      K = input × W_k    # [batch, seq_len, n_embd]
      V = input × W_v    # [batch, seq_len, n_embd]

  Each token becomes a query (asking what to attend to), a key (advertising its content),
  and a value (providing its information).

  ### 2. Multi-Head Split

  We split the embedding dimension into multiple heads for parallel attention:

      n_head = 12, n_embd = 768
      head_dim = n_embd / n_head = 64

      Q: [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_dim]
      K: [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_dim]
      V: [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_dim]

  Each head can learn different attention patterns (syntax, semantics, position, etc.).

  ### 3. Attention Scores

  Compute similarity between all query-key pairs:

      scores = Q × K^T    # [batch, n_head, seq_len, seq_len]

      # Example for 4 tokens:
      #            tok0  tok1  tok2  tok3
      # tok0   [   0.5   0.2   0.1   0.3  ]
      # tok1   [   0.4   0.6   0.2   0.1  ]
      # tok2   [   0.3   0.5   0.7   0.2  ]
      # tok3   [   0.2   0.3   0.4   0.8  ]

  scores[i][j] represents how much token i should attend to token j.

  ### 4. Scaling

      scores = scores / √head_dim

  Prevents dot products from becoming too large, which would cause softmax
  to produce extreme (nearly one-hot) distributions.

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

  ### 7. Apply Attention to Values

  Weighted sum of value vectors:

      output = attention_weights × V    # [batch, n_head, seq_len, head_dim]

      # Each output is a weighted combination:
      # output[tok2] = 0.2×V[tok0] + 0.3×V[tok1] + 0.5×V[tok2]

  This aggregates information from attended tokens into a new representation.

  ### 8. Concatenate Heads

  Merge all heads back together:

      output: [batch, n_head, seq_len, head_dim] → [batch, seq_len, n_embd]

  ### 9. Output Projection

  Final linear transformation:

      output = output × W_proj    # [batch, seq_len, n_embd]

  This allows the model to mix information across heads.

  ## Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)

  Standard multi-head attention: each head has its own Q, K, V projections.

  MQA/GQA optimization: multiple query heads share the same key and value heads.
  This reduces memory usage (smaller KV cache) with minimal quality loss.

      Standard MHA:  n_head = 12, n_kv_head = 12  (no sharing)
      GQA:           n_head = 12, n_kv_head = 4   (3 Q heads per KV head)
      MQA:           n_head = 12, n_kv_head = 1   (all Q heads share 1 KV)

  ## Why Self-Attention Works

  1. **Global context**: Every token can (potentially) attend to every other token
  2. **Dynamic weights**: Attention patterns change based on input content
  3. **Parallel computation**: All positions computed simultaneously (unlike RNNs)
  4. **Interpretable**: Attention weights show what the model focuses on

  ## Computational Complexity

  - Time: O(seq_len² × n_embd) - quadratic in sequence length
  - Memory: O(seq_len² × n_head) for attention matrices

  This is why long sequences are expensive and why alternatives like Mamba exist.

  ## Parameters

  - W_q: [n_embd, n_embd] - Query projection
  - W_k: [n_embd, n_kv_head × head_dim] - Key projection
  - W_v: [n_embd, n_kv_head × head_dim] - Value projection
  - W_proj: [n_embd, n_embd] - Output projection

  Total parameters ≈ 4 × n_embd² (for standard MHA)
  """

  import Nx.Defn

  @doc """
  Creates a causal self-attention layer.

  ## Parameters
    - input: Axon layer [batch, seq_len, n_embd]
    - n_embd: Embedding dimension
    - n_head: Number of attention heads
    - name: Base name for the layer
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
