defmodule NanoAi.LLM.Dataloader do
  @moduledoc """
  Data loading and batching for language model training.

  ## Overview

  The data loader is responsible for:
  1. Reading raw text data from files
  2. Tokenizing text into sequences
  3. Creating input-target pairs for next-token prediction
  4. Batching sequences for efficient training
  5. Shuffling data for better generalization

  ## Data Format for Training

  For causal language modeling, we need input-target pairs:

      Text: "The cat sat on the mat"
      Tokens: [45, 892, 234, 678, 912, 445, <eos>]

      Input:  [45, 892, 234, 678, 912, 445]
      Target: [892, 234, 678, 912, 445, <eos>]

  The model learns to predict each next token given all previous tokens.

  ## Tokenization and Padding

  Each line of text is tokenized using `NanoAi.Tokenizer.encode/1`, which:
  - Truncates sequences longer than 1024 tokens
  - Pads sequences shorter than 1024 tokens with `<|pad|>`
  - Returns uniform-length token sequences

  This ensures all sequences have exactly 1024 tokens before batching.

  ## Input-Target Pairs

  In the loss function the shifting of the inputs and targets happens.

  ## Batching

  Lines are grouped into batches with uniform shapes:

      Batch shape: {batch_size, sequence_length}

      Input batch:  [batch_size, 1024] (s64)
      Target batch: [batch_size, 1024] (s64)

  ## Memory Efficiency

  All sequences are exactly 1024 tokens (handled by tokenizer):
  - No variable-length tensor issues
  - Predictable memory usage: ~4 bytes × batch_size × 1024 per batch
  - Efficient GPU utilization with uniform shapes

  ## Usage

      # From text files (recommended)
      train_data = NanoAi.LLM.DataLoader.from_files("priv/data/train/*/*.txt",
        batch_size: 8,
        sequence_length: 1024
      )

      # From list of texts
      train_data = NanoAi.LLM.DataLoader.from_texts(texts,
        batch_size: 8,
        sequence_length: 1024
      )

      # Use with trainer
      NanoAi.LLM.Trainer.train(model, train_data, epochs: 10)
  """
  alias NanoAi.Tokenizer

  @doc """
  Creates a data stream from text files.

  ## Parameters
    - pattern: Glob pattern for text files (e.g., "priv/data/*.txt")
    - opts: Options

  ## Options
    - :batch_size - Number of sequences per batch. Default: 8
    - :sequence_length - Fixed sequence length. Default: 1024
  """
  @spec from_files(pattern :: String.t(), opts :: keyword()) :: Enumerable.t()
  def from_files(pattern, opts \\ []) do
    with {:ok, tk} <- Tokenizer.load() do
      pattern
      |> Path.wildcard()
      |> Stream.flat_map(&stream_file/1)
      |> from_texts(tk, opts)
    end
  end

  @doc """
  Creates a data stream from a list or stream of texts.

  ## Parameters
    - texts: Enumerable of text strings
    - opts: Options

  ## Options
    - :batch_size - Number of sequences per batch. Default: 8
  """
  @spec from_texts(texts :: Enumerable.t(), tokenizer :: Tokenizers.Tokenizer.t(), opts :: keyword()) :: Enumerable.t()
  def from_texts(texts, tokenizer, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 8)

    texts
    |> Stream.map(&tokenize_for_training(&1, tokenizer))
    |> Stream.filter(&valid_sequence?/1)
    |> Stream.chunk_every(batch_size)
    |> Stream.map(&batch_to_tensors/1)
  end

  defp stream_file(file_path) do
    file_path
    |> File.stream!()
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
  end

  # Tokenize and create input-target pair
  # Tokenizer handles padding/truncation to sequence_length (1024 tokens)
  defp tokenize_for_training(text, tokenizer) do
    case Tokenizer.encode(tokenizer, text) do
      {:ok, %{ids: ids}} ->
        # ids are already 1024 tokens from tokenizer
        # For next-token prediction:
        # input: all tokens except the last one
        # target: all tokens except the first one (shifted left by 1)
        ids
        |> Nx.from_binary(:u32)
        |> Nx.backend_transfer()
        |> then(fn ids -> {ids, ids} end)

      {:error, _} ->
        nil
    end
  end

  # Filter out failed tokenizations
  defp valid_sequence?({_input, _target}), do: true
  defp valid_sequence?(nil), do: false

  # Convert batch to tensors - sequences already uniform length from tokenizer
  # OPTIMIZATION: Ensure proper tensor allocation and deallocation
  defp batch_to_tensors(batch) do
    {input_ids_list, target_ids_list} = Enum.unzip(batch)

    # Create tensors with explicit type for better memory management
    # Using backend directly for more control over memory allocation
    input_tensor = input_ids_list |> Nx.stack() |> Nx.backend_transfer()
    target_tensor = target_ids_list |> Nx.stack() |> Nx.backend_transfer()

    {input_tensor, target_tensor}
  end
end
