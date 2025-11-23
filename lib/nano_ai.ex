defmodule NanoAi do
  @moduledoc """
  Documentation for `NanoAi`.
  """

  alias NanoAi.LLM.Dataloader
  alias NanoAi.LLM.Generator
  alias NanoAi.LLM.Models.GPT
  alias NanoAi.LLM.Trainer

  def train(_opts \\ []) do
    "priv/data/train/*/*.txt"
    |> Dataloader.from_files()
    |> then(fn train_data ->
      Trainer.train(GPT.build(), train_data, epochs: 1)
    end)
  end

  def resume(from_checkpoint, _opts \\ []) do
    "priv/data/train/*/*.txt"
    |> Dataloader.from_files()
    |> then(fn train_data ->
      Trainer.resume(GPT.build(), train_data, from_checkpoint)
    end)
  end

  def generate(prompt, opts \\ []) do
    fn -> {opts[:max_tokens] || 10, prompt} end
    |> Stream.resource(
      fn
        {0, _} ->
          {:halt, :done}

        {max_tokens, prompt} ->
          case Generator.generate(NanoAi.Generator.GPT, prompt, temperature: 2.0) do
            "<|eos|>" ->
              {:halt, :done}

            token ->
              {[token], {max_tokens - 1, prompt <> token}}
          end
      end,
      fn _ -> :ok end
    )
    |> Enum.join()
    |> then(fn result -> Enum.join([prompt, result]) end)
  end

  def explore(%Axon{} = model, template \\ {8, 1024}) do
    Axon.Display.as_table(model, Nx.template(template, :u32))
  end
end
