defmodule NanoAi.Tokenizer do
  @moduledoc """
  Provides tokenization functionalities for text processing in NanoAi.
  """
  alias Tokenizers.Decoder
  alias Tokenizers.Encoding
  alias Tokenizers.Model
  alias Tokenizers.Model.BPE
  alias Tokenizers.PreTokenizer
  alias Tokenizers.Tokenizer
  alias Tokenizers.Trainer

  @pad "<|pad|>"
  @sequence_length Application.compile_env(:nano_ai, __MODULE__)[:sequence_length]
  @special_tokens Application.compile_env(:nano_ai, __MODULE__)[:special_tokens]
  @tk_file_path "priv/data/trained/tokenizer.json"

  def train(files) when is_list(files) do
    with {:ok, %Model{} = bpe} <- model(),
         {:ok, %Tokenizer{} = tokenizer} <- tokenizer(bpe),
         {:ok, %Trainer{} = trainer} <- trainer(),
         {:ok, %Tokenizer{} = tokenizer} <- train(tokenizer, trainer, files) do
      Tokenizer.save(tokenizer, @tk_file_path)
    end
  end

  def load(path \\ @tk_file_path) when is_binary(path) do
    Tokenizer.from_file(path)
  end

  @doc """
  Encodes a single text or a batch of texts into token IDs using the default tokenizer.
  """
  @spec encode(text_or_batch :: String.t() | [String.t(), ...]) ::
          {:error, term()}
          | {:ok, %{ids: binary(), attention_mask: binary()}}
          | {:ok, [%{ids: binary(), attention_mask: binary()}]}
  def encode(text_or_batch) when is_binary(text_or_batch) or is_list(text_or_batch) do
    with {:ok, %Tokenizer{} = tokenizer} <- load() do
      case text_or_batch do
        text when is_binary(text) ->
          encode(tokenizer, text)

        texts when is_list(texts) ->
          encode_batch(tokenizer, texts)
      end
    end
  end

  @doc """
  Encodes a single text into token IDs.
  """
  @spec encode(tokenizer :: Tokenizer.t(), text :: String.t()) ::
          {:error, term()} | {:ok, %{ids: binary(), attention_mask: binary()}}
  def encode(%Tokenizer{} = tokenizer, text) when is_binary(text) do
    with {:ok, %Encoding{} = encoding} <- Tokenizer.encode(tokenizer, text) do
      encoding
      |> Encoding.truncate(@sequence_length)
      |> Encoding.pad(@sequence_length, pad_token: @pad)
      |> then(fn enc ->
        {:ok, %{ids: Encoding.get_u32_ids(enc), attention_mask: Encoding.get_u32_attention_mask(enc)}}
      end)
    end
  end

  @doc """
  Encodes a batch of texts into token IDs.
  """
  @spec encode_batch(tokenizer :: Tokenizer.t(), text :: [String.t(), ...]) ::
          {:error, term()} | {:ok, [%{ids: binary(), attention_mask: binary()}]}
  def encode_batch(%Tokenizer{} = tokenizer, texts) when is_list(texts) do
    with {:ok, [%Encoding{} | _] = encodings} <- Tokenizer.encode_batch(tokenizer, texts) do
      {:ok,
       Enum.map(encodings, fn enc ->
         enc
         |> Encoding.truncate(@sequence_length)
         |> Encoding.pad(@sequence_length, pad_token: @pad)
         |> then(fn enc ->
           %{ids: Encoding.get_u32_ids(enc), attention_mask: Encoding.get_u32_attention_mask(enc)}
         end)
       end)}
    end
  end

  def files do
    ["priv", "data", "train", "*", "*.txt"]
    |> Path.join()
    |> Path.wildcard()
  end

  @split_regex "'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
  defp pre_tokenizer,
    do:
      PreTokenizer.sequence([
        PreTokenizer.split(@split_regex, :isolated, invert: false),
        PreTokenizer.byte_level(add_prefix_space: false, use_regex: false)
      ])

  defp model do
    BPE.init(%{}, [], byte_fallback: true, fuse_unk: false)
  end

  defp trainer do
    Trainer.bpe(
      vocab_size: 65_536,
      min_frequency: 0,
      special_tokens: @special_tokens,
      initial_alphabet: PreTokenizer.byte_level_alphabet(),
      show_progress: true
    )
  end

  defp decoder do
    Decoder.byte_level()
  end

  defp train(%Tokenizer{} = tokenizer, %Trainer{} = trainer, files) do
    Tokenizer.train_from_files(tokenizer, files, trainer: trainer)
  end

  defp tokenizer(%Model{} = model) do
    with {:ok, %Tokenizer{} = tokenizer} <- Tokenizer.init(model) do
      {:ok,
       tokenizer
       |> Tokenizer.set_pre_tokenizer(pre_tokenizer())
       |> Tokenizer.set_decoder(decoder())}
    end
  end
end
