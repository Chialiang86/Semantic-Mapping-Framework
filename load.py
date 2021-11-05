import numpy as np
from PIL import Image
import numpy as np
import cv2
import json
from habitat_sim.utils.common import d3_40_colors_rgb
import habitat_sim


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0.0,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 65535).astype(np.uint16)
    return depth_img

def load_scene_dict():
    with open('info_semantic.json', 'r') as f:
        return json.load(f)
# scene_dict = load_scene_dict()
# cv2.imwrite(semantic_dir, transform_semantic(semantic.astype(np.uint8), scene_dict))
def transform_semantic(semantic_obs, scene_dict):
    instance_id_to_semantic_label_id = np.array(scene_dict["id_to_label"])
    semantic = instance_id_to_semantic_label_id[semantic_obs]
    semantic_img = Image.new("L", (semantic.shape[1], semantic.shape[0]))        
    semantic_img.putdata(semantic.flatten())
    return np.array(semantic_img)

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
FLOOR = 0 # 0 : first, 1 : second
fname_key = {0:'first_floor', 1:'second_floor'}
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, FLOOR, 0.0])  # agent in world space, [0, 0, 0] : first floor, [0, 1, 0] : second floor
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")

scene_dict = load_scene_dict()

def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        cv2.imshow("semantic", transform_semantic(observations["semantic_sensor"], scene_dict))

        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

        return observations["color_sensor"], observations["depth_sensor"], observations["semantic_sensor"], sensor_state
    return None, None, None

def writeState(rgb, depth, semantic, sensor_state, write_cnt):
    # write to folder "Data_collection"
    root = './Data_collection/{}'.format(fname_key[FLOOR])
    rgb_dir = '{}/rgb/{}.png'.format(root, write_cnt)
    depth_dir = '{}/depth/{}.png'.format(root, write_cnt)
    semantic_dir = '{}/semantic/{}.png'.format(root, write_cnt)
    cv2.imwrite(rgb_dir, transform_rgb_bgr(rgb))
    cv2.imwrite(depth_dir, transform_depth(depth))
    cv2.imwrite(semantic_dir, transform_semantic(semantic.astype(np.uint8), scene_dict))
    gt_pose.write('{} {} {} {} {} {} {}\n'.format(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z))
        
write_cnt = 1
write_signal = 1

gt_pose = open('Data_collection/{}/GT_Pose.txt'.format(fname_key[FLOOR]), 'w') # for position/pose record
action = "move_forward"
rgb, depth, semantic, sensor_state = navigateAndSee(action)

while True:
    writeState(rgb, depth, semantic, sensor_state, write_cnt)
    write_cnt += 1

    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        rgb, depth, semantic, sensor_state = navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        rgb, depth, semantic, sensor_state = navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        rgb, depth, semantic, sensor_state = navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        break
    else:
        print("INVALID KEY")
        continue

