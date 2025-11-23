defmodule NanoAi.LLM.Layers.Embedding do
  @moduledoc """
  Multiple embedding layers implementations.
  """
  import Nx.Defn

  @type positional_opts() :: [
          sequence_length: pos_integer(),
          vocab_size: pos_integer(),
          num_embed: pos_integer()
        ]

  @doc """
  Creates a positional embedding layer.
  ## Parameters
    - opts: Options
  ## Options
    - :vocab_size - Size of the vocabulary (required)
    - :num_embed - Dimension of the embeddings (required)
    - :sequence_length - Length of the input sequence (required)
  """
  @spec positional(opts :: positional_opts()) :: Axon.t()
  def positional(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:vocab_size, :num_embed, :sequence_length])

    vocab_size = opts[:vocab_size]
    num_embed = opts[:num_embed]
    sequence_length = opts[:sequence_length]

    "input-ids"
    |> Axon.input(shape: {nil, nil})
    |> then(fn input_ids ->
      {input_ids, Axon.embedding(input_ids, vocab_size, num_embed)}
    end)
    |> then(fn {input_ids, embeddings} ->
      {input_ids, embeddings,
       (&positional_ids_fun/2)
       |> Axon.layer([input_ids], op_name: :position_ids)
       |> Axon.embedding(sequence_length, num_embed)}
    end)
    |> then(fn {_input_ids, token_embeddings, position_embeddings} ->
      Axon.add(token_embeddings, position_embeddings)
    end)
  end

  # Generates positional indices for the input sequence.
  #
  # Creates a tensor of sequential integers [0, 1, 2, ..., seq_len-1] that will be
  # used to look up position embeddings.
  #
  # Parameters:
  # - input_tensor: Input token IDs [batch, seq_len]
  # - _opts: Options (unused, for compatibility with Axon.layer)
  #
  # Returns: Position indices tensor [1, seq_len] containing [0, 1, 2, ..., seq_len-1]
  defnp positional_ids_fun(input_tensor, _opts) do
    {_, seq_len} = Nx.shape(input_tensor)

    Nx.iota({1, seq_len}, axis: 1)
  end
end
