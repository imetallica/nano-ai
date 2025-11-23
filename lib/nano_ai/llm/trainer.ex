defmodule NanoAi.LLM.Trainer do
  @moduledoc """
  Generic training loop for language models.

  ## Overview

  This trainer works with any language model architecture:
  - GPT (decoder-only transformer)
  - Mixture of Experts (MoE) (Soon)
  - Mamba/SSM (Soon)
  - Hybrid architectures (Soon)

  The only requirement is that the model:
  1. Takes input token IDs: [batch, seq_len]
  2. Outputs logits: [batch, seq_len, vocab_size]

  ## Loss Function

  Cross-Entropy Loss for language modeling:

      loss = -1/N × Σ log(P(correct_token))

  For each position, we compute the negative log probability of the correct next token.
  Lower loss = better predictions.

  ## Training Data Format

      Input:  [token_1, token_2, ..., token_n]
      Target: [token_2, token_3, ..., token_n+1]

  The model learns to predict each token given all previous tokens.

  ## Optimizer

  Common choices for LLMs:
  - **AdamW**: Adam with proper weight decay (most common)
  - **Adam**: Adaptive learning rates
  - **SGD**: Simple but requires careful tuning

  AdamW is standard for transformer training.

  ## Learning Rate Schedule

  Typical schedule:
  1. **Warmup**: Gradually increase LR from 0 to peak (e.g., first 1000 steps)
  2. **Cosine decay**: Smoothly decrease LR to minimum

  This helps avoid early training instability.

  ## Key Hyperparameters

  - **Learning rate**: Peak LR (e.g., 3e-4 for small models, 1e-4 for large)
  - **Batch size**: Number of sequences per step (limited by GPU memory)
  - **Weight decay**: L2 regularization strength (e.g., 0.1)
  - **Gradient clipping**: Max gradient norm (e.g., 1.0)
  - **Warmup steps**: Steps to reach peak LR (e.g., 1000)

  ## Training Loop Structure

      for epoch in 1..num_epochs do
        for batch in data do
          # Forward pass
          logits = model(batch.input_ids)

          # Compute loss
          loss = cross_entropy(logits, batch.target_ids)

          # Backward pass + update
          params = optimizer_step(params, gradients(loss))
        end
      end

  ## Metrics to Monitor

  - **Training loss**: Should decrease over time
  - **Validation loss**: Monitor for overfitting
  - **Perplexity**: exp(loss) - interpretable metric
  - **Learning rate**: Track schedule
  - **Gradient norm**: Check for exploding/vanishing gradients

  ## Usage

      # Prepare data
      train_data = NanoAi.LLM.DataLoader.load("train.txt")

      # Build model
      model = NanoAi.LLM.GPT.build()

      # Train
      trained_params = NanoAi.LLM.Trainer.train(model, train_data,
        epochs: 10,
        batch_size: 8,
        learning_rate: 3.0e-4
      )
  """

  import Nx.Defn

  alias Axon.Loop
  alias Axon.Loop.State
  alias Axon.Losses
  alias Polaris.Optimizers

  require Logger

  @checkpoint_path "priv/data/checkpoint"
  @trained_models_path "priv/data/trained/models"

  @doc """
  Trains a language model on the given data.

  ## Parameters
    - model: Axon model (any architecture that outputs logits)
    - train_data: Stream or list of batches
    - opts: Training options

  ## Options
    - :epochs - Number of training epochs. Default: 1
    - :learning_rate - Peak learning rate. Default: 3.0e-4
    - :optimizer - Optimizer to use. Default: :adamw
    - :loss - Loss function. Default: :cross_entropy
    - :weight_decay - AdamW weight decay. Default: 0.1
    - :max_grad_norm - Gradient clipping. Default: 1.0
    - :log_every - Log metrics every N steps. Default: 1
    - :checkpoint_every - Save model every N steps. Default: 100
  """
  def train(model, train_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 1)
    log_every = Keyword.get(opts, :log_every, 1)
    checkpoint_every = Keyword.get(opts, :checkpoint_every, 100)
    iterations = Keyword.get(opts, :iterations, -1)

    optimizer = build_optimizer(opts)
    loss = build_loss(opts)

    model
    |> Loop.trainer(loss, optimizer)
    |> Loop.log(&log_message/1, event: :iteration_completed, filter: [every: log_every])
    |> Loop.checkpoint(event: :iteration_completed, filter: [every: checkpoint_every], path: @checkpoint_path)
    |> Loop.run(train_data, %{},
      epochs: epochs,
      iterations: iterations,
      garbage_collect: true,
      force_garbage_collection?: true
    )
  end

  @doc """
  Resumes training from a serialized state.

  ## Parameters
    - model: Axon model
    - train_data: Training data
    - state_path: Path to serialized state file
    - opts: Training options
  """
  def resume(model, train_data, state_path, opts \\ []) do
    Logger.info("Resuming from checkpoint: #{state_path}.")

    epochs = Keyword.get(opts, :epochs, 1)
    log_every = Keyword.get(opts, :log_every, 100)
    checkpoint_every = Keyword.get(opts, :checkpoint_every, 100)
    iterations = Keyword.get(opts, :iterations, -1)

    optimizer = build_optimizer(opts)
    loss = build_loss(opts)

    state = state_path |> File.read!() |> Loop.deserialize_state()

    model
    |> Loop.trainer(loss, optimizer)
    |> Loop.log(&log_message/1, event: :iteration_completed, filter: [every: log_every])
    |> Loop.checkpoint(event: :iteration_completed, filter: [every: checkpoint_every], path: @checkpoint_path)
    |> Loop.from_state(state)
    |> Loop.run(train_data, %{},
      epochs: epochs,
      iterations: iterations,
      garbage_collect: true,
      force_garbage_collection?: true
    )
  end

  def save(model, name, _opts \\ []) do
    path = Path.join([@trained_models_path, "#{name}.axon"])

    %{
      params: model,
      datetime: DateTime.to_unix(DateTime.utc_now()),
      nx_version: :erlang.phash2(Application.spec(:nx, :vsn))
    }
    |> Nx.serialize()
    |> then(&File.write(path, &1))
    |> tap(fn _ ->
      Logger.info("Model saved at: #{path}.")
    end)
  end

  def load(name, _opts \\ []) do
    path = Path.join([@trained_models_path, "#{name}.axon"])

    with {:ok, contents} <- File.read(path) do
      %{params: model, datetime: datetime, nx_version: nx_version} = Nx.deserialize(contents)
      Logger.info("Loaded model saved at #{name}.")
      {:ok, model}
    end
  end

  def build_optimizer(opts \\ []) do
    optimizer_type = Keyword.get(opts, :optimizer, :adamw)
    learning_rate = Keyword.get(opts, :learning_rate, 3.0e-4)
    weight_decay = Keyword.get(opts, :weight_decay, 0.1)
    max_grad_norm = Keyword.get(opts, :max_grad_norm, 1.0)

    optimizer =
      case optimizer_type do
        :adamw ->
          Optimizers.adamw(learning_rate: learning_rate, decay: weight_decay)

        :adam ->
          Optimizers.adam(learning_rate: learning_rate)

        :sgd ->
          Optimizers.sgd(learning_rate: learning_rate)

        custom when is_function(custom) ->
          custom
      end

    Polaris.Updates.compose(
      Polaris.Updates.clip_by_global_norm(max_norm: max_grad_norm),
      optimizer
    )
  end

  def build_loss(opts \\ []) do
    loss_type = Keyword.get(opts, :loss, :cross_entropy)

    case loss_type do
      :cross_entropy ->
        &cross_entropy_loss/2

      :cross_entropy_with_label_smoothing ->
        smoothing = Keyword.get(opts, :label_smoothing, 0.1)
        &cross_entropy_loss(&1, &2, smoothing: smoothing)

      custom when is_function(custom) ->
        custom
    end
  end

  defnp cross_entropy_loss(y_true, y_pred, _opts \\ []) do
    # y_true: [batch, seq_len]
    # y_pred: [batch, seq_len, vocab_size]
    {_, y_true_dim} = Nx.shape(y_true)
    {_, y_pred_dim, _} = Nx.shape(y_pred)
    y_true_shifted = Nx.slice_along_axis(y_true, 1, y_true_dim - 1, axis: 1)
    y_pred_shifted = Nx.slice_along_axis(y_pred, 0, y_pred_dim - 1, axis: 1)
    y_true_shifted = Nx.new_axis(y_true_shifted, -1)

    Losses.categorical_cross_entropy(y_true_shifted, y_pred_shifted,
      from_logits: true,
      sparse: true,
      reduction: :mean
    )
  end

  defp log_message(%State{metrics: metrics} = state) do
    # Log memory usage if available
    memory_info = get_memory_info()
    loss = Float.round(Nx.to_number(metrics["loss"]), 5)
    perplexity = Float.round(Nx.to_number(Nx.exp(metrics["loss"])), 5)

    "\n[Training] Epoch #{state.epoch}, Step #{state.iteration}, Loss: #{loss}, Perplexity: #{perplexity} | Memory: [#{memory_info}]"
  end

  defp get_memory_info do
    memory_data = :erlang.memory()
    total_mb = div(memory_data[:total], 1_048_576)
    processes_mb = div(memory_data[:processes], 1_048_576)
    "#{total_mb}MB total, #{processes_mb}MB processes"
  end
end
