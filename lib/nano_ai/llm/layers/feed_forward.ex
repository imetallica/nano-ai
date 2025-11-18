defmodule NanoAi.LLM.Layers.FeedForward do
  @moduledoc """
  Feed-Forward Network (FFN) for transformer blocks.

  ## Overview

  The FFN is a simple two-layer neural network applied independently to each token position.
  It provides the transformer with non-linear transformation capacity and is where much of
  the model's "knowledge" is believed to be stored.

  ## Architecture

      Input → Linear (expand) → Activation → Linear (contract) → Output

      [batch, seq_len, n_embd] → [batch, seq_len, 4*n_embd] → [batch, seq_len, n_embd]

  ## The Two Projections

  ### 1. Up Projection (Expansion)

      hidden = input × W_up    # [batch, seq_len, n_embd] → [batch, seq_len, 4*n_embd]

  Expands the representation to a higher dimension (typically 4× the embedding size).
  This gives the network more capacity to learn complex transformations.

  ### 2. Down Projection (Contraction)

      output = activation(hidden) × W_down    # [batch, seq_len, 4*n_embd] → [batch, seq_len, n_embd]

  Projects back to the original dimension after applying non-linearity.

  ## Activation Functions

  Common choices:
  - **ReLU**: Simple, fast, but can have "dead neurons"
  - **GELU**: Smooth approximation of ReLU, used in GPT-2/3, BERT
  - **SiLU/Swish**: x × sigmoid(x), used in LLaMA, PaLM

  GELU is the standard for GPT-like models.

  ## Why 4× Expansion?

  The 4× factor is empirically chosen:
  - Large enough to provide representational capacity
  - Small enough to be computationally tractable
  - Consistent across GPT, BERT, LLaMA, and most transformers

  Some models use different ratios (e.g., 8/3× for LLaMA with SwiGLU).

  ## Position-wise Application

  The FFN is applied **independently** to each token position:

      Token 0: FFN(embedding[0]) → output[0]
      Token 1: FFN(embedding[1]) → output[1]
      Token 2: FFN(embedding[2]) → output[2]
      ...

  No information flows between positions in the FFN - that's the attention layer's job.
  This is why it's called "position-wise" feed-forward.

  ## Role in the Transformer

  The transformer block alternates between:
  1. **Attention**: Mixes information ACROSS positions (communication)
  2. **FFN**: Transforms EACH position independently (computation)

  Think of it as:
  - Attention = "gather relevant information from other tokens"
  - FFN = "process and transform that information"

  ## Parameters

  For n_embd = 768:
  - W_up: [768, 3072] = 2,359,296 parameters
  - W_down: [3072, 768] = 2,359,296 parameters
  - Total: ~4.7M parameters per FFN layer

  In a 12-layer model, FFN accounts for roughly 2/3 of all parameters!

  ## Variants

  ### Standard FFN (GPT-2, BERT)
      hidden = GELU(input × W_up)
      output = hidden × W_down

  ### Gated Linear Unit (GLU) variants (LLaMA, PaLM)
      gate = input × W_gate
      hidden = input × W_up
      output = (gate × SiLU(hidden)) × W_down

  GLU variants add an extra "gate" that controls information flow,
  often improving performance at the cost of more parameters.
  """

  @doc """
  Standard FFN (GPT-2, BERT style). Non gated.

      hidden = activation(input × W_up)
      output = hidden × W_down
  """
  @spec gelu(
          input :: Axon.t(),
          n_embed :: integer(),
          name :: String.t(),
          opts :: keyword()
        ) :: Axon.t()
  def gelu(input, n_embed, name, opts \\ [expand_factor: 4]) do
    hidden_size = n_embed * opts[:expand_factor]

    input
    |> up_projection(hidden_size, name)
    |> activation_layer(:gelu, name)
    |> down_projection(n_embed, name)
  end

  @doc """
  Gated Linear Unit FFN (LLaMA, PaLM style).

      gate = SiLU(input × W_gate)
      hidden = input × W_up
      output = (gate * hidden) × W_down

  Uses SwiGLU activation by default.
  """
  @spec gated(
          input :: Axon.t(),
          n_embed :: integer(),
          name :: String.t(),
          opts :: keyword()
        ) :: Axon.t()
  def gated(input, n_embed, name, opts \\ [expand_factor: 4]) do
    hidden_size = n_embed * opts[:expand_factor]

    gate =
      input
      |> Axon.dense(hidden_size, use_bias: false, name: "#{name}.gate-projection")
      |> activation_layer(:silu, "#{name}.gate")

    up = up_projection(input, hidden_size, name)

    gate
    |> Axon.multiply(up, name: "#{name}.gated-multiply")
    |> down_projection(n_embed, name)
  end

  @doc """
  ReGLU variant (ReLU-gated).

      gate = ReLU(input × W_gate)
      hidden = input × W_up
      output = (gate * hidden) × W_down
  """
  @spec reglu(
          input :: Axon.t(),
          n_embed :: integer(),
          name :: String.t(),
          opts :: keyword()
        ) :: Axon.t()
  def reglu(input, n_embed, name, opts \\ [expand_factor: 4]) do
    gated(input, n_embed, name, Keyword.put(opts, :activation, :relu))
  end

  @doc """
  GEGLU variant (GELU-gated).

      gate = GELU(input × W_gate)
      hidden = input × W_up
      output = (gate * hidden) × W_down
  """
  @spec geglu(
          input :: Axon.t(),
          n_embed :: integer(),
          name :: String.t(),
          opts :: keyword()
        ) :: Axon.t()
  def geglu(input, n_embed, name, opts \\ [expand_factor: 4]) do
    gated(input, n_embed, name, Keyword.put(opts, :activation, :gelu))
  end

  defp up_projection(input, hidden_size, name) do
    Axon.dense(input, hidden_size, use_bias: false, name: "#{name}.up-projection")
  end

  defp activation_layer(input, activation, name) do
    case activation do
      :gelu -> Axon.gelu(input, name: "#{name}.activation.gelu")
      :silu -> Axon.silu(input, name: "#{name}.activation.silu")
      :relu -> Axon.relu(input, name: "#{name}.activation.relu")
    end
  end

  defp down_projection(input, num_embed, name) do
    Axon.dense(input, num_embed, use_bias: false, name: "#{name}.down-projection")
  end
end
