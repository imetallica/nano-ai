defmodule NanoAi.TokenizerTest do
  use ExUnit.Case, async: true

  alias NanoAi.Tokenizer

  require Logger

  @moduletag :tokenizer

  describe "Tokenizer.encode/1" do
    test "encodes a single text" do
      text = "Hello, world!"
      {:ok, token_ids} = Tokenizer.encode(text)
      assert is_list(token_ids)
      assert Enum.all?(token_ids, &is_integer/1)
    end

    test "encodes a batch of texts" do
      texts = ["Hello, world!", "How are you?"]
      {:ok, batch_token_ids} = Tokenizer.encode(texts)
      assert is_list(batch_token_ids)
      assert length(batch_token_ids) == length(texts)

      Enum.each(batch_token_ids, fn token_ids ->
        assert is_list(token_ids)
        assert Enum.all?(token_ids, &is_integer/1)
      end)
    end
  end
end
