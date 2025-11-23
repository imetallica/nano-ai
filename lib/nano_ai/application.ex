defmodule NanoAi.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Starts a worker by calling: NanoAi.Worker.start_link(arg)
      # {Nx.Serving, name: NanoAi.Generator.GPT, serving: load_serving("trained-model-file-name")}
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: NanoAi.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # defp load_serving(file_path) do
  #   with {:ok, state} <- NanoAi.LLM.Trainer.load(file_path) do
  #     NanoAi.LLM.Models.GPT.serving(state)
  #   end
  # end
end
