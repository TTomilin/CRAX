from typing import List, Dict, Any, Optional
import os
import xml.etree.ElementTree as ET

from brax.envs.goals import GoalManager
from brax.envs.hazards import HazardManager


# Navigation type compatibility mapping
NAVIGATION_COMPATIBILITY = {
    "2d": ["2d"],  # 2D agents can only do 2D tasks
    "1d_forward": ["1d_forward"],  # 1D forward agents can only do 1D tasks
}


def get_assets_dir() -> str:
    """Get the path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_agent_xml_path(agent_name: str) -> str:
    """Get the path to an agent XML file."""
    return os.path.join(get_assets_dir(), "agents", f"{agent_name}.xml")


def get_task_xml_path(task_name: str) -> str:
    """Get the path to a task XML file."""
    return os.path.join(get_assets_dir(), "tasks", f"{task_name}.xml")


class ModularXMLBuilder:
    """Builds complete XML environments by combining modular agent and task XMLs."""

    def __init__(self, model_name: str = "brax_environment"):
        """Initialize the modular XML builder.

        Args:
            model_name: Name of the MuJoCo model
        """
        self.model_name = model_name

    def _parse_xml(self, xml_path: str) -> ET.Element:
        """Parse an XML file and return the root element."""
        tree = ET.parse(xml_path)
        return tree.getroot()

    def _element_to_string(self, element: ET.Element, indent: int = 0) -> str:
        """Convert an XML element to a formatted string."""
        # Simple serialization that preserves attributes
        indent_str = "  " * indent
        tag = element.tag
        attrs = " ".join(f'{k}="{v}"' for k, v in element.attrib.items())

        if len(element) == 0 and element.text is None:
            # Self-closing tag
            if attrs:
                return f"{indent_str}<{tag} {attrs}/>"
            return f"{indent_str}<{tag}/>"

        # Opening tag
        if attrs:
            result = f"{indent_str}<{tag} {attrs}>"
        else:
            result = f"{indent_str}<{tag}>"

        # Content
        if element.text and element.text.strip():
            result += element.text.strip()

        # Child elements
        has_children = len(element) > 0
        if has_children:
            result += "\n"
            for child in element:
                result += self._element_to_string(child, indent + 1) + "\n"
            result += f"{indent_str}</{tag}>"
        else:
            result += f"</{tag}>"

        return result

    def _get_navigation_type(self, root: ET.Element) -> str:
        """Extract navigation type from an agent or task XML."""
        return root.attrib.get("navigation", "unknown")

    def check_compatibility(self, agent_path: str, task_path: str) -> bool:
        """Check if an agent is compatible with a task based on navigation type.

        Args:
            agent_path: Path to agent XML
            task_path: Path to task XML

        Returns:
            True if compatible, False otherwise
        """
        agent_root = self._parse_xml(agent_path)
        task_root = self._parse_xml(task_path)

        agent_nav = self._get_navigation_type(agent_root)
        task_nav = self._get_navigation_type(task_root)

        compatible_navs = NAVIGATION_COMPATIBILITY.get(agent_nav, [])
        return task_nav in compatible_navs

    def build_combined_xml(
        self,
        agent_name: str,
        task_name: str,
        goal_manager: Optional[GoalManager] = None,
        hazard_manager: Optional[HazardManager] = None,
        agent_position: tuple = None,
    ) -> str:
        """Build a complete XML by combining an agent with a task.

        Args:
            agent_name: Name of the agent (e.g., "point", "ant", "walker2d")
            task_name: Name of the task (e.g., "goal_navigation", "forward_run")
            goal_manager: Optional GoalManager for dynamic goals
            hazard_manager: Optional HazardManager for dynamic hazards
            agent_position: Optional (x, y, z) position to place the agent

        Returns:
            Complete MuJoCo XML string
        """
        agent_path = get_agent_xml_path(agent_name)
        task_path = get_task_xml_path(task_name)

        # Check compatibility
        if not self.check_compatibility(agent_path, task_path):
            agent_root = self._parse_xml(agent_path)
            task_root = self._parse_xml(task_path)
            agent_nav = self._get_navigation_type(agent_root)
            task_nav = self._get_navigation_type(task_root)
            raise ValueError(
                f"Agent '{agent_name}' (navigation={agent_nav}) is not compatible "
                f"with task '{task_name}' (navigation={task_nav})"
            )

        agent_root = self._parse_xml(agent_path)
        task_root = self._parse_xml(task_path)

        # Start building the combined XML
        xml_parts = [f'<mujoco model="{self.model_name}">']

        # Extract and merge compiler settings (prefer agent's if exists)
        agent_compiler = agent_root.find("compiler")
        task_compiler = task_root.find("compiler")
        if agent_compiler is not None:
            xml_parts.append(self._element_to_string(agent_compiler, 1))
        elif task_compiler is not None:
            xml_parts.append(self._element_to_string(task_compiler, 1))

        # Extract and merge size settings (prefer task's if exists)
        task_size = task_root.find("size")
        if task_size is not None:
            xml_parts.append(self._element_to_string(task_size, 1))

        # Extract and merge option settings (prefer agent's for physics params)
        agent_option = agent_root.find("option")
        task_option = task_root.find("option")
        if agent_option is not None:
            xml_parts.append(self._element_to_string(agent_option, 1))
        elif task_option is not None:
            xml_parts.append(self._element_to_string(task_option, 1))

        # Extract and merge visual settings (prefer task's)
        task_visual = task_root.find("visual")
        if task_visual is not None:
            xml_parts.append(self._element_to_string(task_visual, 1))

        # Extract and merge statistic settings (prefer task's)
        task_stat = task_root.find("statistic")
        if task_stat is not None:
            xml_parts.append(self._element_to_string(task_stat, 1))

        # Extract and merge default settings (prefer agent's)
        agent_default = agent_root.find("default")
        if agent_default is not None:
            xml_parts.append(self._element_to_string(agent_default, 1))

        # Extract and merge custom settings (combine both)
        xml_parts.append("  <custom>")
        agent_custom = agent_root.find("custom")
        task_custom = task_root.find("custom")
        if agent_custom is not None:
            xml_parts.append("    <!-- Agent custom params -->")
            for child in agent_custom:
                xml_parts.append(self._element_to_string(child, 2))
        if task_custom is not None:
            xml_parts.append("    <!-- Task custom params -->")
            for child in task_custom:
                xml_parts.append(self._element_to_string(child, 2))
        xml_parts.append("  </custom>")

        # Merge assets (from both task and managers)
        xml_parts.append("  <asset>")
        task_asset = task_root.find("asset")
        if task_asset is not None:
            for child in task_asset:
                xml_parts.append(self._element_to_string(child, 2))
        if goal_manager and goal_manager.get_goal_count() > 0:
            goal_assets = goal_manager.get_xml_assets()
            if goal_assets:
                xml_parts.append(f"    {goal_assets}")
        if hazard_manager and hazard_manager.get_hazard_count() > 0:
            hazard_assets = hazard_manager.get_xml_assets()
            if hazard_assets:
                xml_parts.append(f"    {hazard_assets}")
        xml_parts.append("  </asset>")

        # Build worldbody
        xml_parts.append("  <worldbody>")

        # Add task worldbody elements (floor, lights, cameras, static objects)
        task_worldbody = task_root.find("worldbody")
        if task_worldbody is not None:
            for child in task_worldbody:
                # Skip the AGENT_PLACEHOLDER comment
                if child.tag == "body" and child.attrib.get("name") == "agent":
                    continue  # We'll add the agent body separately
                xml_parts.append(self._element_to_string(child, 2))

        # Add agent body
        agent_body = agent_root.find("body")
        if agent_body is not None:
            # Optionally override agent position
            if agent_position is not None:
                agent_body.set("pos", f"{agent_position[0]} {agent_position[1]} {agent_position[2]}")
            xml_parts.append("    <!-- Agent body -->")
            xml_parts.append(self._element_to_string(agent_body, 2))

        # Add dynamic goals
        if goal_manager and goal_manager.get_goal_count() > 0:
            xml_parts.append("    <!-- Goals -->")
            xml_parts.extend(goal_manager.get_xml_bodies())

        # Add dynamic hazards
        if hazard_manager and hazard_manager.get_hazard_count() > 0:
            xml_parts.append("    <!-- Hazards -->")
            xml_parts.extend(hazard_manager.get_xml_bodies())

        xml_parts.append("  </worldbody>")

        # Add agent sensors
        agent_sensor = agent_root.find("sensor")
        if agent_sensor is not None:
            xml_parts.append(self._element_to_string(agent_sensor, 1))

        # Add agent actuators
        agent_actuator = agent_root.find("actuator")
        if agent_actuator is not None:
            xml_parts.append(self._element_to_string(agent_actuator, 1))

        xml_parts.append("</mujoco>")

        return "\n".join(xml_parts)

    @staticmethod
    def list_available_agents() -> List[str]:
        """List all available agent XMLs."""
        agents_dir = os.path.join(get_assets_dir(), "agents")
        if not os.path.exists(agents_dir):
            return []
        return [f[:-4] for f in os.listdir(agents_dir) if f.endswith(".xml")]

    @staticmethod
    def list_available_tasks() -> List[str]:
        """List all available task XMLs."""
        tasks_dir = os.path.join(get_assets_dir(), "tasks")
        if not os.path.exists(tasks_dir):
            return []
        return [f[:-4] for f in os.listdir(tasks_dir) if f.endswith(".xml")]

    @staticmethod
    def get_compatible_agents(task_name: str) -> List[str]:
        """Get list of agents compatible with a given task."""
        task_path = get_task_xml_path(task_name)
        if not os.path.exists(task_path):
            return []

        task_root = ET.parse(task_path).getroot()
        task_nav = task_root.attrib.get("navigation", "unknown")

        compatible = []
        for agent_name in ModularXMLBuilder.list_available_agents():
            agent_path = get_agent_xml_path(agent_name)
            agent_root = ET.parse(agent_path).getroot()
            agent_nav = agent_root.attrib.get("navigation", "unknown")

            if task_nav in NAVIGATION_COMPATIBILITY.get(agent_nav, []):
                compatible.append(agent_name)

        return compatible

    @staticmethod
    def get_compatible_tasks(agent_name: str) -> List[str]:
        """Get list of tasks compatible with a given agent."""
        agent_path = get_agent_xml_path(agent_name)
        if not os.path.exists(agent_path):
            return []

        agent_root = ET.parse(agent_path).getroot()
        agent_nav = agent_root.attrib.get("navigation", "unknown")

        compatible = []
        for task_name in ModularXMLBuilder.list_available_tasks():
            task_path = get_task_xml_path(task_name)
            task_root = ET.parse(task_path).getroot()
            task_nav = task_root.attrib.get("navigation", "unknown")

            if task_nav in NAVIGATION_COMPATIBILITY.get(agent_nav, []):
                compatible.append(task_name)

        return compatible


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

