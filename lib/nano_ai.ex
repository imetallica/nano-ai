defmodule NanoAi do
  @moduledoc """
  Documentation for `NanoAi`.
  """

  alias NanoAi.LLM.Dataloader
  alias NanoAi.LLM.GPT
  alias NanoAi.LLM.Trainer

  def train(_opts \\ []) do
    "priv/data/train/*/*.txt"
    |> Dataloader.from_files()
    |> then(fn train_data ->
      Trainer.train(GPT.build(), train_data, epochs: 10)
    end)
  end
end
