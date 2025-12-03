import inspect
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Tuple

import jax.numpy as jp


class BaseHazard(ABC):
    """Base class for all hazard types."""

    def __init__(self, hazard_id: int, position: tuple, size: Tuple[float] | float, height: float, collidable: bool,
                 fixed: bool, density: float):
        """Initialize a hazard.

        Args:
            hazard_id: Unique identifier for this hazard
            position: (x, y, z) position of the hazard
            size: Size parameter for the hazard (interpretation depends on hazard type)
            height: Height parameter for the hazard (if applicable)
            collidable: Whether the hazard is collidable
            fixed: Whether the hazard should be randomly relocated on reset
            density: Density of the hazard (if movable)
        """
        self.hazard_id = hazard_id
        self.position = position
        self.size = size
        self.height = height
        self.collidable = collidable
        self.fixed = fixed
        self.density = density
        self.mass = self.calculate_mass()
        self.geom_id = -1  # Will be populated by the environment after mj_model is created.
        self.contype = self.conaffinity = 1 if collidable else 0
        self.alpha = 0.8 if collidable else 0.35

    def calculate_cost(self,
                       agent_xy: jp.ndarray,
                       hazard_xy: jp.ndarray,
                       cost_scaler: float = 1.0,
                       contact_geom1: jp.ndarray | None = None,
                       contact_geom2: jp.ndarray | None = None,
                       contact_dist: jp.ndarray | None = None,
                       ncon: jp.ndarray | None = None,
                       agent_geom_ids: jp.ndarray | None = None) -> jp.ndarray:
        """Common wrapper:
        - If collidable and contact buffers are provided: binary 1.0 on any contact.
        - Else: fall back to subclass proximity_cost.
        """
        if self.collidable and (contact_geom1 is not None) and (agent_geom_ids is not None):
            return cost_scaler * self._collision_binary_cost(
                contact_geom1, contact_geom2, contact_dist, ncon, agent_geom_ids
            )
        # Proximity shaping for non-collidable or when contacts aren’t available
        return cost_scaler * self.proximity_cost(agent_xy, hazard_xy)

    def _collision_binary_cost(self,
                               contact_geom1: jp.ndarray,
                               contact_geom2: jp.ndarray,
                               contact_dist: jp.ndarray,
                               ncon: jp.ndarray,
                               agent_geom_ids: jp.ndarray) -> jp.ndarray:
        """Returns 1.0 if any agent↔this-hazard contact (dist <= 0), else 0.0."""
        max_slots = contact_geom1.shape[0]
        valid = (jp.arange(max_slots) < ncon)

        is_agent1 = (contact_geom1[None, :] == agent_geom_ids[:, None]).any(axis=0)
        is_agent2 = (contact_geom2[None, :] == agent_geom_ids[:, None]).any(axis=0)
        is_haz1 = (contact_geom1 == self.geom_id)
        is_haz2 = (contact_geom2 == self.geom_id)

        pair = (is_agent1 & is_haz2) | (is_haz1 & is_agent2)
        touch = contact_dist <= 0.0
        any_contact = jp.any(valid & pair & touch)
        return jp.where(any_contact, 1.0, 0.0)

    @abstractmethod
    def proximity_cost(self, agent_xy: jp.ndarray, hazard_xy: jp.ndarray) -> jp.ndarray:
        """Distance/AABB-based shaping for NON-collidable usage."""
        raise NotImplementedError

    @abstractmethod
    def get_xml_body(self) -> str:
        """Generate XML body definition for this hazard.

        Returns:
            XML string defining the hazard body
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hazard_type(self) -> str:
        """Return the type name of this hazard."""
        raise NotImplementedError

    @abstractmethod
    def calculate_mass(self) -> float:
        """Return mass in kilograms computed from geometry (size/height) and density."""
        raise NotImplementedError

    @abstractmethod
    def get_keepout_radius(self) -> float:
        """Return the keepout radius for placement constraints."""
        raise NotImplementedError

    @abstractmethod
    def get_keepout_shape(self):
        """
        Return a shape descriptor for placement:
          - ("circle", jp.array([r]))
          - ("rect",   jp.array([sx, sy]))   # half-extents in x/y
        """
        raise NotImplementedError


class CubeHazard(BaseHazard):
    """Cube-shaped hazard with binary collision cost."""

    def __init__(self, hazard_id: int, position: tuple = (0.0, 0.0, 0.09), size: float = 0.2, height: float = 0.2,
                 collidable: bool = True, fixed: bool = False, density: float = 1.0):
        super().__init__(hazard_id, position, size, height, collidable, fixed, density)

    def proximity_cost(self, agent_xy: jp.ndarray, hazard_xy: jp.ndarray) -> jp.ndarray:
        dxdy = jp.abs(agent_xy - hazard_xy)
        inside = jp.logical_and(dxdy[0] <= self.size, dxdy[1] <= self.size)
        return inside.astype(jp.float32)

    def get_xml_body(self) -> str:
        """Generate XML body for cube hazard."""
        x, y, z = self.position

        return f"""
        <body name="hazard{self.hazard_id}" pos="{x} {y} {z}" mocap="true">
            <geom type="box" name="hazard{self.hazard_id}" size="{self.size} {self.size} {self.height}" condim="3"
                  friction="1 .03 .003" rgba="0.9 0.3 0.3 {self.alpha}" contype="{self.contype}" 
                  conaffinity="{self.conaffinity}" mass="{self.mass}" solref="0.01 1"/>
        </body>"""

    @property
    def hazard_type(self) -> str:
        return "cube"

    def calculate_mass(self) -> float:
        # Geom "box" size is half-extents (sx, sy, sz). Here `size` is the half-extent in x/y.
        sx = sy = float(self.size)
        sz = float(self.height)
        volume = (2.0 * sx) * (2.0 * sy) * (2.0 * sz)
        rho = float(self.density)
        return rho * volume

    def get_keepout_radius(self) -> float:
        # box uses half-size in xy: safe radius is circumscribed
        return float(jp.sqrt(2.0) * self.size)

    def get_keepout_shape(self):
        # square AABB with half-extent = self.size
        return "rect", jp.array([float(self.size), float(self.size)])


class RectHazard(BaseHazard):
    """
    Axis-aligned rectangle hazard (box geom) with independent half-extents in x/y.
    Interprets `size` as a tuple `(sx, sy)` of half-extents in XY.
    """
    def __init__(self,
                 hazard_id: int,
                 position: tuple = (0.0, 0.0, 0.02),
                 size: tuple = (0.5, 0.05),  # (sx, sy)
                 height: float = 0.02,
                 collidable: bool = False,
                 fixed: bool = False,
                 density: float = 1.0):
        # normalize size to tuple
        if not (isinstance(size, (tuple, list)) and len(size) == 2):
            raise ValueError("RectHazard.size must be a (sx, sy) tuple of half-extents.")
        self.size_xy = (float(size[0]), float(size[1]))
        super().__init__(hazard_id, position, size, height, collidable, fixed, density)

    def proximity_cost(self, agent_xy: jp.ndarray, hazard_xy: jp.ndarray) -> jp.ndarray:
        sx, sy = self.size_xy
        dxdy = jp.abs(agent_xy - hazard_xy)
        inside = jp.logical_and(dxdy[0] <= sx, dxdy[1] <= sy)
        return inside.astype(jp.float32)

    def get_xml_body(self) -> str:
        x, y, z = self.position
        sx, sy = self.size_xy
        sz = float(self.height)
        return f"""
        <body name="hazard{self.hazard_id}" pos="{x} {y} {z}" mocap="true">
            <geom type="box" name="hazard{self.hazard_id}" size="{sx} {sy} {sz}" condim="3"
                  friction="1 .03 .003" rgba="0.9 0.3 0.3 {self.alpha}" contype="{self.contype}"
                  conaffinity="{self.conaffinity}" mass="{self.mass}" solref="0.01 1"/>
        </body>"""

    @property
    def hazard_type(self) -> str:
        return "rect"

    def calculate_mass(self) -> float:
        sx, sy = self.size_xy
        sz = float(self.height)
        volume = (2.0 * sx) * (2.0 * sy) * (2.0 * sz)
        return float(self.density) * volume

    def get_keepout_radius(self) -> float:
        sx, sy = self.size_xy
        return float(jp.sqrt(sx * sx + sy * sy))  # circumscribed radius

    def get_keepout_shape(self):
        sx, sy = self.size_xy
        return "rect", jp.array([float(sx), float(sy)])


class CylinderHazard(BaseHazard):
    """Cylinder-shaped hazard with distance-based cost."""

    def __init__(self, hazard_id: int, position: tuple = (0.0, 0.0, 0.02), size: float = 0.3, height: float = 0.02,
                 collidable: bool = True, fixed: bool = False, density: float = 1.0):
        super().__init__(hazard_id, position, size, height, collidable, fixed, density)

    def proximity_cost(self, agent_xy: jp.ndarray, hazard_xy: jp.ndarray) -> jp.ndarray:
        diff = agent_xy - hazard_xy
        dist = jp.sqrt(jp.sum(diff * diff) + 1e-8)
        return jp.maximum(0.0, 1.0 - dist / self.size)

    def get_xml_body(self) -> str:
        """Generate XML body for cylinder hazard."""
        x, y, z = self.position

        return f'''    
        <body name="hazard{self.hazard_id}" pos="{x} {y} {z}" mocap="true">
            <geom type="cylinder" name="hazard{self.hazard_id}" size="{self.size} {self.height}" 
                rgba="0.9 0.3 0.3 {self.alpha}" contype="{self.contype}" conaffinity="{self.conaffinity}" 
                mass="{self.mass}" solref="0.01 1"/>
        </body>'''

    @property
    def hazard_type(self) -> str:
        return "cylinder"

    def calculate_mass(self) -> float:
        r = float(self.size)
        h = float(self.height)
        volume = float(jp.pi) * r * r * h
        rho = float(self.density)
        return rho * volume

    def get_keepout_radius(self) -> float:
        return self.size  # Radius

    def get_keepout_shape(self):
        return "circle", jp.array([float(self.size)])


class HazardManager:
    """Manages a collection of hazards and generates XML."""

    def __init__(self):
        self.hazards: List[BaseHazard] = []

    def add_hazard(self, hazard: BaseHazard):
        """Add a hazard to the manager."""
        self.hazards.append(hazard)

    def add_hazards(self, hazard_type: str, count: int, positions: List[tuple] = None, size: float = None,
                    height: float = None, collidable: bool = None, fixed: bool = False, density: float = None):
        """Add multiple hazards of the same type.

        Args:
            hazard_type: "cube" or "cylinder"
            count: Number of hazards to add
            positions: List of (x, y, z) positions. If None, default positions will be used.
            size: Size parameter. If None, default size will be used.
        """
        cls = HAZARD_REGISTRY.get(hazard_type)
        if cls is None:
            available = ", ".join(sorted(set(HAZARD_REGISTRY.keys())))
            raise ValueError(f"Unknown hazard type '{hazard_type}'. Available: {available}")

        if positions is None:
            positions = [(0.0, 0.0, 0.0)] * count

        for i in range(count):
            hazard_id = len(self.hazards) + 1
            hazard = cls(hazard_id, positions[i], size, height, collidable, fixed, density)
            self.add_hazard(hazard)

    def get_xml_assets(self) -> str:
        """Generate XML asset definitions for hazards.

        Returns:
            XML string defining hazard materials
        """
        assets = [
            '<material name="hazard" rgba="0.9 0.3 0.3 1"/>',
            '<material name="hazard_cylinder" rgba="0.9 0.3 0.3 0.5"/>'
        ]
        return '\n    '.join(assets)

    def get_xml_bodies(self) -> List[str]:
        """Generate XML body definitions for all hazards.

        Returns:
            List of XML strings, one for each hazard body
        """
        bodies = []
        for hazard in self.hazards:
            bodies.append(hazard.get_xml_body())
        return bodies

    def clear(self):
        """Remove all hazards."""
        self.hazards.clear()

    def get_hazard_count(self) -> int:
        """Get the total number of hazards."""
        return len(self.hazards)

    def get_fixed_hazard_count(self) -> int:
        """Get the total number of hazards."""
        return sum(1 for h in self.hazards if h.fixed)

    def get_hazards_by_type(self, hazard_type: str) -> List[BaseHazard]:
        """Get all hazards of a specific type."""
        return [h for h in self.hazards if h.hazard_type == hazard_type]


HAZARD_REGISTRY: Dict[str, Type[BaseHazard]] = {
    "cube": CubeHazard,
    "cylinder": CylinderHazard,
    "rect": RectHazard,
}


def _type_defaults_from_registry():
    d = {}
    for t, cls in HAZARD_REGISTRY.items():
        sig = inspect.signature(cls.__init__)

        def get(name):
            p = sig.parameters.get(name)
            return None if p is None or p.default is inspect._empty else p.default

        # pull whatever the class exposes; no hardcoding
        base = {}
        for k in ("size", "height", "collidable", "movable", "density"):
            v = get(k)
            if v is not None:
                base[k] = v
        d[t] = base
    return d
