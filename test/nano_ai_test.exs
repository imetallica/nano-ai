defmodule NanoAiTest do
  use ExUnit.Case
  doctest NanoAi

  test "greets the world" do
    assert NanoAi.hello() == :world
  end
end
