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
# This file is derived from Brax 0.12.3 (training/agents/ppo/__init__.py) and has been modified
# by the CRAX Authors: re-exports the CRAX `ppo_cost` trainer alongside `train`.

from . import train
from .ppo_cost import RewardMinusCostWrapper, train_ppo_cost

__all__ = [
    "train",
    "RewardMinusCostWrapper",
    "train_ppo_cost",
]

