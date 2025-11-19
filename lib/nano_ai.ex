defmodule NanoAi do
  @moduledoc """
  Documentation for `NanoAi`.
  """

  alias NanoAi.LLM.Dataloader
  alias NanoAi.LLM.Models.GPT
  alias NanoAi.LLM.Trainer

  def train(_opts \\ []) do
    "priv/data/train/*/*.txt"
    |> Dataloader.from_files()
    |> then(fn train_data ->
      Trainer.train(GPT.build(), train_data, epochs: 1)
    end)
  end

  def explore(%Axon{} = model, template \\ {8, 1024}) do
    Axon.Display.as_table(model, Nx.template(template, :bf16))
  end
end
