"""Command-line entrypoints for the asteroid RL demo.

Each submodule is runnable as ``python -m asteroid_rl.cli.<name>``. Modules in
this package parse CLI arguments, construct the environment or policies, and
delegate to ``asteroid_rl.env``, ``asteroid_rl.policies``, and
``asteroid_rl.episode``.
"""
