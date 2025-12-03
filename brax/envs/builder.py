from typing import List, Dict, Any, Optional

from brax.envs.goals import GoalManager
from brax.envs.hazards import HazardManager


class XMLBuilder:
    """Builds complete XML environments from components (agent, goals, hazards)."""

    def __init__(self, model_name: str = "brax_environment"):
        """Initialize the XML builder.

        Args:
            model_name: Name of the MuJoCo model
        """
        self.model_name = model_name

    def build_xml(self,
                  agent_xml_path: str,
                  goal_manager: Optional[GoalManager] = None,
                  hazard_manager: Optional[HazardManager] = None,
                  additional_assets: List[str] = None,
                  additional_bodies: List[str] = None) -> str:
        """Build complete XML from components.

        Args:
            agent_xml_path: Path to the base agent XML file
            goal_manager: GoalManager containing goals
            hazard_manager: HazardManager containing hazards
            additional_assets: Additional asset definitions
            additional_bodies: Additional body definitions

        Returns:
            Complete XML string
        """
        xml_parts = [
            f'<mujoco model="{self.model_name}">',
            f'  <include file="{agent_xml_path}"/>',
            ''
        ]

        # Collect assets from all components
        assets = []
        if goal_manager and goal_manager.get_goal_count() > 0:
            goal_assets = goal_manager.get_xml_assets()
            if goal_assets:
                assets.append(goal_assets)

        if hazard_manager and hazard_manager.get_hazard_count() > 0:
            hazard_assets = hazard_manager.get_xml_assets()
            if hazard_assets:
                assets.append(hazard_assets)

        if additional_assets:
            assets.extend(additional_assets)

        if assets:
            xml_parts.extend([
                '  <asset>',
                *[f'    {asset}' for asset in assets],
                '  </asset>',
                ''
            ])

        # Start worldbody
        xml_parts.append('  <worldbody>')

        # Add goals
        if goal_manager and goal_manager.get_goal_count() > 0:
            xml_parts.extend([
                '      <!-- Goals -->',
                *goal_manager.get_xml_bodies(),
                ''
            ])

        # Add hazards
        if hazard_manager and hazard_manager.get_hazard_count() > 0:
            xml_parts.extend([
                '      <!-- Hazards -->',
                *hazard_manager.get_xml_bodies(),
                ''
            ])

        # Add additional bodies
        if additional_bodies:
            xml_parts.extend([
                '      <!-- Additional Bodies -->',
                *additional_bodies,
                ''
            ])

        # Close worldbody and mujoco
        xml_parts.extend([
            '  </worldbody>',
            '</mujoco>'
        ])

        return '\n'.join(xml_parts)


class EnvironmentBuilder:
    """High-level builder for creating complete environments."""

    def __init__(self):
        self.xml_builder = XMLBuilder()

    def create_environment_spec(self,
                                agent_type: str = "point",
                                goal_specs: List[Dict] = None,
                                hazard_specs: List[Dict] = None,
                                model_name: str = None) -> Dict[str, Any]:
        """Create an environment specification.

        Args:
            agent_type: Type of agent ("point", "ant", etc.)
            goal_specs: List of goal specifications. Each spec should have:
                       {'type': 'cube'/'cylinder', 'count': int, 'positions': List[tuple] (optional), 'size': Any (optional)}
            hazard_specs: List of hazard specifications. Each spec should have:
                         {'type': 'cube'/'cylinder', 'count': int, 'positions': List[tuple] (optional), 'size': float (optional)}
            model_name: Name for the MuJoCo model

        Returns:
            Dictionary containing the environment specification
        """
        # Create goal manager
        goal_manager = GoalManager()
        if goal_specs:
            for spec in goal_specs:
                gtype = spec.get('type', 'cube')
                count = spec.get('count', 1)
                positions = spec.get('positions', None)
                size = spec.get('size', None)
                goal_manager.add_goals(gtype, count, positions, size)

        # Create hazard manager
        hazard_manager = HazardManager()
        if hazard_specs:
            for spec in hazard_specs:
                htype = spec.get('type', 'cube')
                count = spec.get('count', 1)
                positions = spec.get('positions', None)
                size = spec.get('size', None)
                hazard_manager.add_hazards(htype, count, positions, size)

        # Determine agent XML path
        agent_xml_path = f"{agent_type}.xml"

        # Set model name
        if model_name is None:
            model_name = f"{agent_type}_environment"

        self.xml_builder.model_name = model_name

        return {
            'agent_xml_path': agent_xml_path,
            'goal_manager': goal_manager,
            'hazard_manager': hazard_manager,
            'model_name': model_name
        }

    def build_environment_xml(self, env_spec: Dict[str, Any]) -> str:
        """Build XML from an environment specification.

        Args:
            env_spec: Environment specification from create_environment_spec

        Returns:
            Complete XML string
        """
        return self.xml_builder.build_xml(
            agent_xml_path=env_spec['agent_xml_path'],
            goal_manager=env_spec['goal_manager'],
            hazard_manager=env_spec['hazard_manager']
        )

    def create_point_goal_hazard_environment(self,
                                             goal_specs: List[Dict] = None,
                                             hazard_specs: List[Dict] = None) -> str:
        """Convenience method to create a point agent environment with goals and hazards.

        Args:
            goal_specs: List of goal specifications
            hazard_specs: List of hazard specifications

        Returns:
            Complete XML string
        """
        # Default to single cube goal if none specified
        if goal_specs is None:
            goal_specs = [{'type': 'cube', 'count': 1}]

        # Default to some hazards if none specified
        if hazard_specs is None:
            hazard_specs = [{'type': 'cube', 'count': 4}]

        env_spec = self.create_environment_spec(
            agent_type="point",
            goal_specs=goal_specs,
            hazard_specs=hazard_specs,
            model_name="point_goal_hazard"
        )

        return self.build_environment_xml(env_spec)

