import os

from brax.envs.SafePointGoal import default_config
from brax.envs.env_utils import create_hazard_manager_from_config, create_goal_manager_from_config, \
    generate_point_goal_xml


def test_print_xml_mixed_hazards():
    """Test printing XML configuration with mixed hazards."""
    print("\nGenerating XML with mixed hazards...")

    cfg = default_config()
    cfg.goals.type = 'cylinder'
    cfg.goals.count = 2
    cfg.hazards.specs = [
        # 1) 3 cube hazards, size 0.5, height 0.5, collidable
        {"type": "cube", "count": 3, "size": 0.5, "height": 0.5, "collidable": True},

        # 2) 2 cube hazards, size 0.25, height 2.0, collidable
        {"type": "cube", "count": 2, "size": 0.25, "height": 2.0, "collidable": True},

        # 3) 4 cylinder hazards, radius 0.3, "infinitesimal" height, non-collidable
        {"type": "cylinder", "count": 4, "size": 0.3, "height": 1e-3, "collidable": False},
    ]

    # Create managers
    goal_manager = create_goal_manager_from_config(cfg.goals)
    hazard_manager = create_hazard_manager_from_config(cfg.hazards)

    # Generate XML
    xml_path = generate_point_goal_xml(goal_manager, hazard_manager)

    try:
        # Read and print the XML content
        with open(xml_path, 'r') as f:
            xml_content = f.read()

        print("\n" + "=" * 60)
        print("Generated XML Configuration:")
        print("=" * 60)
        print(xml_content)
        print("=" * 60)

        # Print summary
        cube_hazards = hazard_manager.get_hazards_by_type("cube")
        cylinder_hazards = hazard_manager.get_hazards_by_type("cylinder")
        goal_count = goal_manager.get_goal_count()
        hazard_count = hazard_manager.get_hazard_count()
        print(f"\n✓ XML generated with:")
        print(f"  - {goal_count} goal(s)")
        print(f"  - {hazard_count} total hazards ({len(cube_hazards)} cubes, {len(cylinder_hazards)} cylinders)")

    finally:
        # Clean up temporary XML file
        if os.path.exists(xml_path):
            os.unlink(xml_path)
            print(f"✓ Temporary XML file cleaned up")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    test_print_xml_mixed_hazards()
    print("\n" + "=" * 60)
    print("✓ XML configuration printed successfully.")
