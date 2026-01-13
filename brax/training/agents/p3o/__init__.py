# Copyright 2024 The Brax Authors.
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

"""P3O (Penalized Proximal Policy Optimization) training algorithm.

Reference: Zhang et al., "Penalized Proximal Policy Optimization for Safe
Reinforcement Learning", IJCAI 2022.
https://arxiv.org/abs/2205.11814
"""

from brax.training.agents.p3o.train import train
from brax.training.agents.p3o import networks
from brax.training.agents.p3o import losses
