defmodule NanoAi.LLM.GPT do
  @moduledoc """
  GPT-like language model built with Axon.

  ## Overview

  This module implements a decoder-only transformer language model similar to GPT-2/3.
  The model predicts the next token given a sequence of previous tokens.

  ## Architecture

      Input Token IDs: [batch, seq_len]
        ↓
      Token Embeddings: [batch, seq_len, num_embed]
        +
      Position Embeddings: [batch, seq_len, num_embed]
        ↓
      Transformer Block 1
        ↓
      Transformer Block 2
        ↓
      ...
        ↓
      Transformer Block N (num_layers)
        ↓
      Final Layer Norm
        ↓
      Output Projection (to vocab_size)
        ↓
      Logits: [batch, seq_len, vocab_size]

  ## Configuration

  Using NanoAi config:
  - sequence_length: 1024 (context window)
  - vocab_size: 65,536 (tokenizer vocabulary)
  - num_layers: 12 (transformer blocks)
  - num_heads: 6 (attention heads)
  - num_embed: 768 (embedding dimension)

  ## Total Parameters

  Token Embeddings: 65,536 × 768 = 50,331,648 (~50M)
  Position Embeddings: 1,024 × 768 = 786,432 (~0.8M)
  Transformer Blocks: 12 × ~7.1M = ~85M
  Final LayerNorm: 768 × 2 = 1,536
  Output Projection: 768 × 65,536 = 50,331,648 (~50M)

  Total: ~186M parameters

  Note: Many implementations tie the output projection weights to the token
  embeddings (weight tying), which would reduce this by ~50M parameters.

  ## Training Objective

  Causal Language Modeling (next-token prediction):
  - Input: [token_1, token_2, ..., token_n]
  - Target: [token_2, token_3, ..., token_n+1]
  - Loss: Cross-entropy between predicted logits and target tokens

  ## Inference

  Autoregressive generation:
  1. Encode prompt tokens
  2. Forward pass to get logits
  3. Sample next token from logits
  4. Append to sequence
  5. Repeat until <|eos|> or max_length

  ## Usage

      # Build the model
      model = NanoAi.LLM.GPT.build()

      # Initialize parameters
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 1024}, :s64), %{})

      # Forward pass
      logits = predict_fn.(params, input_ids)
  """
  import Nx.Defn

  alias NanoAi.LLM.Layers.Transformer

  @config Application.compile_env(:nano_ai, __MODULE__)

  @spec build(opts :: keyword()) :: Axon.t()
  def build(opts \\ @config) do
    opts
    |> embeddings()
    |> transformer_blocks(opts)
    |> final_norm()
    |> output_projection(opts)
  end

  @doc """
  Builds the embedding layer for the GPT model.

  This includes both token embeddings and positional embeddings.

  Why add positions?
    Without position info, the model sees "The cat sat" the same as "sat cat The" - just a bag of tokens.
    Position embeddings tell it that "The" is at position 0, "cat" at position 1, etc.
  """
  def embeddings(opts \\ @config) do
    "input-ids"
    |> Axon.input(shape: {nil, nil})
    |> then(fn input_ids ->
      {input_ids, Axon.embedding(input_ids, opts[:vocab_size], opts[:num_embed], name: "token-embeddings")}
    end)
    |> then(fn {input_ids, embeddings} ->
      {input_ids, embeddings,
       (&positional_ids_fun/2)
       |> Axon.layer([input_ids], name: "position-ids", op_name: :position_ids)
       |> Axon.embedding(opts[:sequence_length], opts[:num_embed], name: "position-embeddings")}
    end)
    |> then(fn {_input_ids, token_embeddings, position_embeddings} ->
      Axon.add(token_embeddings, position_embeddings, name: "embeddings")
    end)
  end

  defnp positional_ids_fun(input_tensor, _opts) do
    {_, seq_len} = Nx.shape(input_tensor)

    Nx.iota({1, seq_len}, axis: 1)
  end

  defp transformer_blocks(input, opts) do
    Enum.reduce(1..opts[:num_layers], input, fn layer_idx, input ->
      case opts[:ffn_norm] do
        :pre_norm ->
          Transformer.pre_norm(input, opts[:num_embed], opts[:num_heads], "pre-transformer-block-#{layer_idx - 1}",
            ffn_type: opts[:ffn_type],
            expand_factor: opts[:ffn_expand_factor]
          )

        :post_norm ->
          Transformer.post_norm(input, opts[:num_embed], opts[:num_heads], "post-transformer-block-#{layer_idx - 1}",
            ffn_type: opts[:ffn_type],
            expand_factor: opts[:ffn_expand_factor]
          )
      end
    end)
  end

  defp final_norm(input) do
    Axon.layer_norm(input, name: "final-layer-norm")
  end

  defp output_projection(input, opts) do
    Axon.dense(input, opts[:vocab_size], use_bias: false, name: "output-head")
  end
end
