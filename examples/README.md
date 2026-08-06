"""Basilisk `bskExamples` dump kept as optional asset fallback.

The project’s real MuJoCo/Itokawa assets live under ``assets/``. ``env``/``gym_env``
only falls back to ``examples/mujoco`` and ``examples/dataForExamples`` if those
files are missing from ``assets/``.

Do not treat this tree as application code. Prefer regenerating via::

    bskExamples

(or reinstalling ``bsk[all,examples]``) rather than editing scenarios here.
"""
