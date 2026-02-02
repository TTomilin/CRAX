"""Safe Spider environment with leg-lifting constraints.

A 6-legged spider that must learn to walk while keeping certain legs off the ground.

Difficulty levels:
  - Level 1: Keep 2 legs up (front-left + back-right diagonal)
  - Level 2: Keep 3 legs up (front-left + mid-right + back-left - alternating tripod)
  - Level 3: Keep 4 legs up (only mid-left + mid-right may touch - center legs only)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np
from jax import numpy as jp

from brax import base
from brax.envs.safe_lift import SafeLift


class SafeLiftSpider(SafeLift):
    """Spider that must keep certain legs off the ground.

    The spider receives a cost each timestep when a restricted leg touches the ground.
    Ground contact is detected using MuJoCo's native contact detection system.
    """

    # Indicator geom names for visualization (transparent red spheres)
    INDICATOR_GEOMS = {
        'front_left': 'indicator_fl',
        'front_right': 'indicator_fr',
        'mid_left': 'indicator_ml',
        'mid_right': 'indicator_mr',
        'back_left': 'indicator_bl',
        'back_right': 'indicator_br',
    }

    @property
    def agent_xml_path(self) -> str:
        return 'envs/assets/safe/spider.xml'

    @property
    def foot_geoms(self) -> Dict[str, Tuple[str, Tuple[float, float, float]]]:
        return {
            'front_left': ('foot_fl_geom', (0.3, 0.3, 0.0)),
            'front_right': ('foot_fr_geom', (-0.3, 0.3, 0.0)),
            'mid_left': ('foot_ml_geom', (0.35, 0.0, 0.0)),
            'mid_right': ('foot_mr_geom', (-0.35, 0.0, 0.0)),
            'back_left': ('foot_bl_geom', (0.3, -0.3, 0.0)),
            'back_right': ('foot_br_geom', (-0.3, -0.3, 0.0)),
        }

    def get_restricted_feet(self, difficulty: int) -> List[str]:
        if difficulty == 1:
            # Level 1: Keep 2 legs up (front-left + back-right diagonal)
            return ['front_left', 'back_right']
        elif difficulty == 2:
            # Level 2: Keep 3 legs up (alternating tripod)
            return ['front_left', 'mid_right', 'back_left']
        else:  # difficulty == 3
            # Level 3: Keep 4 legs up (only center legs may touch)
            return ['front_left', 'front_right', 'back_left', 'back_right']

    @property
    def default_healthy_z_range(self) -> Tuple[float, float]:
        return (0.2, 1.0)

    @property
    def uses_contact_detection(self) -> bool:
        return True  # Uses MuJoCo contact detection

    def __init__(self, difficulty: int = 1, **kwargs):
        super().__init__(difficulty=difficulty, **kwargs)

        # Get indicator geom IDs for visualization (loaded after super().__init__)
        from etils import epath
        path = epath.resource_path('brax') / self.agent_xml_path
        mj_model = mujoco.MjModel.from_xml_path(str(path))

        self._indicator_geom_ids = {}
        for foot_key, indicator_name in self.INDICATOR_GEOMS.items():
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, indicator_name)
            if gid >= 0:
                self._indicator_geom_ids[foot_key] = gid

    def render(
            self,
            trajectory: Union[List[base.State], base.State],
            height: int = 240,
            width: int = 320,
            camera: Optional[str] = None,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """Renders trajectory with violation indicators shown as red spheres.

        When a restricted foot touches the ground, its indicator sphere becomes
        visible (red, semi-transparent) to visualize the constraint violation.
        """
        renderer = mujoco.Renderer(self.sys.mj_model, height=height, width=width)
        camera = camera or -1

        # Store original indicator colors to restore after rendering
        original_rgba = {}
        for foot_key in self._restricted_feet:
            if foot_key in self._indicator_geom_ids:
                indicator_id = self._indicator_geom_ids[foot_key]
                original_rgba[indicator_id] = self.sys.mj_model.geom_rgba[indicator_id].copy()

        def get_image(state: base.State):
            # Check which restricted feet are violating (touching floor)
            feet_touching = self._check_foot_floor_contacts_render(state)

            # Update indicator colors based on violations
            for i, foot_key in enumerate(self._restricted_feet):
                if foot_key not in self._indicator_geom_ids:
                    continue
                indicator_id = self._indicator_geom_ids[foot_key]
                if feet_touching[i]:
                    # Violation: show a red semi-transparent sphere
                    self.sys.mj_model.geom_rgba[indicator_id] = [1.0, 0.0, 0.0, 0.5]
                else:
                    # No violation: keep the sphere transparent
                    self.sys.mj_model.geom_rgba[indicator_id] = [1.0, 0.0, 0.0, 0.0]

            # Render the frame
            d = mujoco.MjData(self.sys.mj_model)
            d.qpos, d.qvel = np.asarray(state.q), np.asarray(state.qd)
            if hasattr(state, 'mocap_pos') and hasattr(state, 'mocap_quat'):
                d.mocap_pos, d.mocap_quat = state.mocap_pos, state.mocap_quat
            mujoco.mj_forward(self.sys.mj_model, d)
            renderer.update_scene(d, camera=camera)
            return renderer.render()

        if isinstance(trajectory, list):
            images = [get_image(s) for s in trajectory]
        else:
            images = get_image(trajectory)

        # Restore original colors
        for indicator_id, rgba in original_rgba.items():
            self.sys.mj_model.geom_rgba[indicator_id] = rgba

        return images

    def _check_foot_floor_contacts_render(self, state: base.State) -> List[bool]:
        """Check foot contacts for rendering."""
        contact_geom = np.asarray(state.contact.geom)
        contact_dist = np.asarray(state.contact.dist)
        active_contacts = contact_dist <= 0

        contacts = []
        for foot_key in self._restricted_feet:
            if foot_key not in self._foot_geom_ids:
                contacts.append(False)
                continue
            foot_geom_id = self._foot_geom_ids[foot_key]

            is_foot_floor_contact = (
                ((contact_geom[:, 0] == foot_geom_id) & (contact_geom[:, 1] == self._floor_geom_id)) |
                ((contact_geom[:, 1] == foot_geom_id) & (contact_geom[:, 0] == self._floor_geom_id))
            )
            in_contact = np.any(is_foot_floor_contact & active_contacts)
            contacts.append(bool(in_contact))

        return contacts
