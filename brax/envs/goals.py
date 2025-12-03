from jax import numpy as jp
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, NamedTuple


class PackedGoalParams(NamedTuple):
    type_id: int  # 0=cube, 1=cylinder, etc.
    radius: float  # used by cylinders; 0 otherwise
    half_extents_xy: tuple[float, float]  # used by cubes; (0,0) otherwise
    yaw: float  # rotation around Z if cubes have orientation; 0 otherwise


class BaseGoal(ABC):
    """Base class for all goal types."""

    def __init__(self, goal_id: int, position: tuple, size: float, height):
        """Initialize a goal.

        Args:
            goal_id: Unique identifier for this goal
            position: (x, y, z) position of the goal
            size: Size parameter for the goal (interpretation depends on goal type)
            height: Height parameter for the goal (if applicable)
        """
        self.goal_id = goal_id
        self.position = position
        self.size = size
        self.height = height

    @abstractmethod
    def get_xml_body(self) -> str:
        """Generate XML body definition for this goal.

        Returns:
            XML string defining the goal body
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def goal_type(self) -> str:
        """Return the type name of this goal."""
        raise NotImplementedError

    @property
    @abstractmethod
    def type_id(self) -> int:
        """Return the id of the goal type."""
        raise NotImplementedError

    @abstractmethod
    def encode_static_params(self) -> PackedGoalParams:
        raise NotImplementedError

    @abstractmethod
    def get_keepout_radius(self) -> float:
        """Return the keepout radius for placement constraints."""
        raise NotImplementedError

    def update_position(self, new_position: tuple):
        """Update the goal's position (for respawning)."""
        self.position = new_position


class CubeGoal(BaseGoal):
    """Cube-shaped goal."""

    def __init__(self, goal_id: int, position: tuple = (0.0, 0.0, 0.0), size: float | tuple = 0.2, height: float = 0.2):
        """Initialize a cube goal.

        Args:
            goal_id: Unique identifier for this goal
            position: (x, y, z) position of the goal
            size: (width, height, depth) size of the goal box
        """
        # For cube goals, size can be a tuple (w, h, d) or a single value
        if isinstance(size, (int, float)):
            size = (size, size)

        super().__init__(goal_id, position, max(size), height)  # Use max dimension for base size
        self.dimensions = size

    def get_xml_body(self) -> str:
        """Generate XML body for cube goal."""
        x, y, z = self.position
        w, h = self.dimensions
        return f'''    
        <body name="goal{self.goal_id}" pos="{x} {y} {z}" mocap="true">
            <geom type="box" name="goal{self.goal_id}" size="{w} {h} {self.height}" condim="3"
                friction="1 .03 .003" rgba="0 1 0 0.8" contype="0" conaffinity="0" solref="0.01 1"/>
        </body>'''

    @property
    def goal_type(self) -> str:
        return "cube"

    @property
    def type_id(self) -> int:
        return 0

    def get_keepout_radius(self) -> float:
        # half-extents in xy
        hx, hy = self.dimensions[0], self.dimensions[1]
        # use circumscribed circle so rotation can’t sneak overlap
        return float(jp.sqrt(hx * hx + hy * hy))

    def encode_static_params(self) -> PackedGoalParams:
        hx, hy = (float(self.dimensions[0]), float(self.dimensions[1]))
        return PackedGoalParams(
            type_id=self.type_id,
            radius=0.0,
            half_extents_xy=(hx, hy),
            yaw=0.0,
        )


class CylinderGoal(BaseGoal):
    """Cylinder-shaped goal."""

    def __init__(self, goal_id: int, position: tuple = (0.0, 0.0, 0.0), size: float = 0.3, height: float = 0.2):
        """Initialize a cylinder goal.

        Args:
            goal_id: Unique identifier for this goal
            position: (x, y, z) position of the goal
            size: Radius of the cylinder
            height: Height of the cylinder
        """
        super().__init__(goal_id, position, size, height)

    def get_xml_body(self) -> str:
        """Generate XML body for cylinder goal."""
        x, y, z = self.position
        return f'''    
        <body name="goal{self.goal_id}" pos="{x} {y} {z}" mocap="true">
            <geom type="cylinder" name="goal{self.goal_id}" size="{self.size} {self.height}" 
                rgba="0 1 0 0.5" contype="0" conaffinity="0" solref="0.01 1"/>
        </body>'''

    @property
    def goal_type(self) -> str:
        return "cylinder"

    @property
    def type_id(self) -> int:
        return 1

    def get_keepout_radius(self) -> float:
        return self.size  # radius

    def encode_static_params(self) -> PackedGoalParams:
        return PackedGoalParams(
            type_id=self.type_id,
            radius=float(self.size),
            half_extents_xy=(0.0, 0.0),
            yaw=0.0,
        )


class GoalManager:
    """Manages a collection of goals and handles goal logic."""

    def __init__(self):
        self.goals: List[BaseGoal] = []

    def add_goal(self, goal: BaseGoal):
        """Add a goal to the manager."""
        self.goals.append(goal)

    def add_goals(self, goal_type: str, count: int, positions: List[tuple] = None, size: Any = None,
                  height: float = None):
        """Add multiple goals of the same type.

        Args:
            goal_type: "cube" or "cylinder"
            count: Number of goals to add
            positions: List of (x, y, z) positions. If None, default positions will be used.
            size: Size parameter. If None, default size will be used.
        """
        cls = GOAL_REGISTRY.get(goal_type)
        if cls is None:
            available = ", ".join(sorted(set(GOAL_REGISTRY.keys())))
            raise ValueError(f"Unknown goal type '{goal_type}'. Available: {available}")

        if positions is None:
            positions = [(0.0, 0.0, 0.0)] * count

        for i in range(count):
            goal_id = len(self.goals) + 1
            goal = cls(goal_id, positions[i], size, height)
            self.add_goal(goal)

    def get_xml_assets(self) -> str:
        """Generate XML asset definitions for goals.

        Returns:
            XML string defining goal materials
        """
        assets = [
            '<material name="goal" rgba="0.3 0.9 0.3 0.7"/>',
            '<material name="goal_inactive" rgba="0.5 0.5 0.5 0.3"/>'
        ]
        return '\n    '.join(assets)

    def get_xml_bodies(self) -> List[str]:
        """Generate XML body definitions for all goals.

        Returns:
            List of XML strings, one for each goal body
        """
        bodies = []
        for goal in self.goals:
            bodies.append(goal.get_xml_body())
        return bodies

    def clear(self):
        """Remove all goals."""
        self.goals.clear()

    def get_goal_count(self) -> int:
        """Get the total number of goals."""
        return len(self.goals)

    def get_goals_by_type(self, goal_type: str) -> List[BaseGoal]:
        """Get all goals of a specific type."""
        return [g for g in self.goals if g.goal_type == goal_type]


GOAL_REGISTRY: Dict[str, Type[BaseGoal]] = {
    "cube": CubeGoal,
    "cylinder": CylinderGoal,
}
