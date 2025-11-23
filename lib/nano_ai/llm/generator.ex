defmodule NanoAi.LLM.Generator do
  @moduledoc false
  alias Nx.Serving

  require Nx

  def generate(serving_name, prompt, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 1.0)

    Serving.batched_run(serving_name, {prompt, temperature})
  end
end
