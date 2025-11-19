defmodule NanoAi.MixProject do
  use Mix.Project

  def project do
    [
      app: :nano_ai,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {NanoAi.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "== 0.8.0"},
      {:nx, "== 0.10.0"},
      {:polaris, "== 0.1.0"},
      {:tokenizers, "== 0.5.1"},

      # API tools
      {:bandit, "== 1.8.0"},
      {:plug, "== 1.18.1"},

      # Coding style
      {:styler, ">= 0.0.0", only: [:dev, :test], runtime: false, optional: true},
      {:table_rex, "== 4.1.0", only: [:dev, :test], optional: true},

      # Specific for Mac, select other versions as needed
      {:emlx, "== 0.2.0", optional: true}
    ]
  end
end
