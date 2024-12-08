import os
import shutil
import yaml

workspace_drake = "/workspace/drake"

def build_base_sdf(from_scratch=False):
    base_dir = f"{workspace_drake}/data"
    obj_dir = os.path.join(base_dir, "obj")
    yaml_path = os.path.join(base_dir, "final_scenario_draft.yaml")

    if from_scratch:
        if os.path.exists(obj_dir):
            shutil.rmtree(obj_dir)
        if os.path.exists(yaml_path):
            os.remove(yaml_path)

    os.makedirs(obj_dir, exist_ok=True)

    class RpyWrapper:
        def __init__(self, deg):
            self.deg = deg
    def rpy_representer(dumper, data):
        return dumper.represent_mapping("!Rpy", {"deg": data.deg})
    
    class TagWrapper:
        def __init__(self, tag, value):
            self.tag = tag
            self.value = value
    def no_quotes_representer(dumper, data):
        return dumper.represent_mapping(data.tag, data.value)

    yaml.add_representer(RpyWrapper, rpy_representer)
    yaml.add_representer(TagWrapper, no_quotes_representer)

    directives = [
        {
            "add_model": {
                "name": "iiwa",
                "file": "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf",
                "default_joint_positions": {
                    "iiwa_joint_1": [-1.57],
                    "iiwa_joint_2": [0.1],
                    "iiwa_joint_3": [0],
                    "iiwa_joint_4": [-1.2],
                    "iiwa_joint_5": [0],
                    "iiwa_joint_6": [1.6],
                    "iiwa_joint_7": [0]
                }
            }
        },
        {
            "add_weld": {
                "parent": "world",
                "child": "iiwa::iiwa_link_0",
                "X_PC": {
                    "translation": [0, 0, 0.5],
                    "rotation": RpyWrapper([0, 0, 180])
                }
            }
        },
        {
            "add_model": {
                "name": "wsg",
                "file": "package://manipulation/hydro/schunk_wsg_50_with_tip.sdf"
            }
        },
        {
            "add_weld": {
                "parent": "iiwa::iiwa_link_7",
                "child": "wsg::body",
                "X_PC": {
                    "translation": [0, 0, 0.09],
                    "rotation": RpyWrapper([90, 0, 90])
                }
            }
        },
        {
            "add_model": {
                "name": "table",
                "file": f"file://{workspace_drake}/table.sdf",
                "default_free_body_pose": {
                    "link": {
                        "translation": [0, 0.25, 0],
                        "rotation": RpyWrapper([0, 0, 0])
                    }
                }
            }
        }
    ]
    
    model_drivers = {
        "model_drivers": {
            "iiwa": TagWrapper("!IiwaDriver", {
                "control_mode": "position_only",
                "hand_model_name": "wsg"
            }),
            "wsg": TagWrapper("!SchunkWsgDriver", {})
        }
    }

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump({"directives": directives, **model_drivers}, yaml_file, default_flow_style=False)

    return yaml_path

if __name__ == "__main__":
    build_base_sdf(from_scratch=True)
