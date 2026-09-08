# Copyright 2024 The Brax Authors.
# Modifications Copyright 2026 CRAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is derived from Brax 0.12.3 (training/__init__.py) and has been modified
# by the CRAX Authors: package docstring; this package was moved out of `brax.training` to a top-level
# `training` package so that `crax` carries no RL dependencies.


"""Constrained RL algorithms and training infrastructure for CRAX.

This package is deliberately kept separate from `crax`, which contains only the
simulation and environment code. `crax` never imports `training`, so the
environments can be used without the RL dependency stack (optax, orbax, wandb);
install those with the `train` extra:

    pip install -e ".[train]"
"""
