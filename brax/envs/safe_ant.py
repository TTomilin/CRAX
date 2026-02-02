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

"""Safe Ant environment with leg-lifting constraints.

The ant must learn to walk while keeping certain legs off the ground.

Difficulty levels:
  - Level 1: Keep front-left leg off the ground
  - Level 2: Keep front-left and back-right (diagonal) legs off the ground
  - Level 3: Only back-right leg may touch the ground (hop on one leg)
"""

from typing import Dict, List, Tuple

from brax.envs.safe_lift import SafeLift


class SafeLiftAnt(SafeLift):
    """Ant that must keep certain legs off the ground.

    The ant receives a cost each timestep when a restricted leg touches the ground.
    Ground contact is detected by computing the world z-position of foot geoms
    and checking if they are below a threshold.
    """

    @property
    def agent_xml_path(self) -> str:
        return 'envs/assets/safe/ant.xml'

    @property
    def foot_geoms(self) -> Dict[str, Tuple[str, Tuple[float, float, float]]]:
        return {
            'front_left': ('left_foot_geom', (0.4, 0.4, 0.0)),
            'front_right': ('right_foot_geom', (-0.4, 0.4, 0.0)),
            'back_left': ('third_foot_geom', (-0.4, -0.4, 0.0)),
            'back_right': ('fourth_foot_geom', (0.4, -0.4, 0.0)),
        }

    def get_restricted_feet(self, difficulty: int) -> List[str]:
        if difficulty == 1:
            # Level 1: Front-left leg must stay off ground
            return ['front_left']
        elif difficulty == 2:
            # Level 2: Diagonal legs (front-left and back-right) must stay off
            return ['front_left', 'back_right']
        else:  # difficulty == 3
            # Level 3: Only back-right can touch (all others restricted)
            return ['front_left', 'front_right', 'back_left']

    @property
    def default_healthy_z_range(self) -> Tuple[float, float]:
        return (0.2, 1.0)

    @property
    def uses_contact_detection(self) -> bool:
        return False  # Uses z-position based detection

    def __init__(self, difficulty: int = 1, **kwargs):
        assert difficulty in [1, 2, 3], "Difficulty must be 1, 2, or 3"
        super().__init__(difficulty=difficulty, **kwargs)
