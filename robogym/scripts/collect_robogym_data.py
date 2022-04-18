import logging

import click
import numpy as np
from robogym.envs.rearrange.common.base import RearrangeEnv, VisionArgs, GoalArgs

from robogym.envs.rearrange.common.utils import _is_valid_proposal
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal
NUM_OBJECTS = 4


# def my_sample_object_colors(self, num_groups):
#     all_colors = [[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,0.5,0,1],[1,0,0.5,1],[0.5,0,1,1],[0,0.5,1,1],[0.5,1,0,1],[0,1,0.5,1],[0.25,0,0.5,1]]
#     # select a random color from the list of colors
#     random_color_ids = self._random_state.randint(0,len(all_colors),num_groups)
#     random_colors = np.array([all_colors[i] for i in random_color_ids])
#     return random_colors

# def my_sample_random_object_groups(
#         self, dedupe_objects):
#         return RearrangeEnv._sample_random_object_groups(dedupe_objects=True)

# def _my_is_valid_proposal():
#     print('testing')

def my_place_objects_in_grid(
    object_bounding_boxes,
    table_dimensions,
    placement_area,
    random_state,
    max_num_trials = 5, initial_placements=[]):
    """
    Place objects within rectangular boundaries by dividing the placement area into a grid of cells
    of equal size, and then randomly sampling cells for each object to be placed in.

    :param object_bounding_boxes: matrix of bounding boxes (num_objects, 2, 3) where [:, 0, :]
        contains the center position of the bounding box in Cartesian space relative to the body's
        frame of reference and where [:, 1, :] contains the half-width, half-height, and half-depth
        of the object.
    :param table_dimensions: Tuple (table_pos, table_size, table_height) defining dimension of
        the table where
            table_pos: position of the table.
            table_size: half-size of the table along (x, y, z).
            table_height: height of the table.
    :param placement_area: the placement area in which to place objects.
    :param random_state: numpy random state to use to shuffle placement positions
    :param max_num_trials: maximum number of trials to run (a trial will fail if there is overlap
        detected between any two placements; generally this shouldn't happen with this algorithm)
    :return: Tuple[np.ndarray, bool], where the array is of size (num_objects, 3) with columns set
        to the x, y, z coordinates of objects relative to the world frame, and the boolean
        indicates whether the placement is valid.
    """
    offset_x, offset_y, _ = placement_area.offset
    width, height, _ = placement_area.size
    table_pos, table_size, table_height = table_dimensions

    def _get_global_placement(placement: np.ndarray):
        return placement + [offset_x, offset_y, 0.0] - table_size + table_pos

    # 1. Determine the number of rows and columns of the grid, based on the largest object width
    # and height.
    total_object_area = 0.0
    n_objects = object_bounding_boxes.shape[0]
    max_obj_height = 0.0
    max_obj_width = 0.0
    for i in range(n_objects):
        # Bounding boxes are in half-sizes.
        obj_width = object_bounding_boxes[i, 1, 0] * 2
        obj_height = object_bounding_boxes[i, 1, 1] * 2

        max_obj_height = max(max_obj_height, obj_height)
        max_obj_width = max(max_obj_width, obj_width)

        object_area = obj_width * obj_height
        total_object_area += object_area

    n_columns = int(width // max_obj_width)
    n_rows = int(height // max_obj_height)
    n_cells = n_columns * n_rows

    cell_width = width / n_columns
    cell_height = height / n_rows

    if n_cells < n_objects:
        # Cannot find a valid placement via this method; give up.
        logging.warning(
            f"Unable to fit {n_objects} objects into placement area with {n_cells} cells"
        )
        return np.zeros(shape=(n_objects, 3)), False

    for trial_i in range(max_num_trials):
        placement_valid = True
        placements = [i for i in initial_placements]

        # 2. Initialize an array with all valid cell coordinates.

        # Create an array of shape (n_rows, n_columns, 2) where each element contains the row,col
        # coord
        coords = np.dstack(np.mgrid[0:n_rows, 0:n_columns])
        # Create a shuffled list where ever entry is a valid (row, column) coordinate.
        coords = np.reshape(coords, (n_rows * n_columns, 2))
        random_state.shuffle(coords)
        coords = list(coords)

        # 3. Place each object into a randomly selected cell.
        object_bounding_boxes_including_initial = np.vstack([object_bounding_boxes, object_bounding_boxes])
        for object_idx in range(n_objects):
            row, col = coords.pop()
            pos, size = object_bounding_boxes[object_idx]

            prop_x = cell_width * col + size[0] - pos[0]
            prop_y = cell_height * row + size[1] - pos[1]

            # Reference is to (xmin, ymin, zmin) of table.
            prop_z = object_bounding_boxes[object_idx, 1, -1] + 2 * table_size[-1]
            prop_z -= object_bounding_boxes[object_idx, 0, -1]

            placement = _get_global_placement(np.array([prop_x, prop_y, prop_z]))

            b1_x, b1_y = placement[:2]
            
            if not _is_valid_proposal(
                b1_x, b1_y, object_idx, object_bounding_boxes_including_initial, placements
            ):
                placement_valid = False
                logging.warning(f"Trial {trial_i} failed on object {object_idx}")
                break

            placements.append(placement)
        placements = placements[len(initial_placements):]
        if placement_valid:
            assert (
                len(placements) == n_objects
            ), "There should be a placement for every object"
            break

    return np.array(placements), placement_valid

def my_sample_next_goal_positions(self, random_state):
    initial_placements = []
    for step in range(NUM_OBJECTS):
        if np.all(self.mujoco_simulation.qpos[8+7*step:8+7*step+3]!=0):
            initial_placements.append(self.mujoco_simulation.qpos[8+7*step:8+7*step+3])
    placement, is_valid = my_place_objects_in_grid(
        self.mujoco_simulation.get_object_bounding_boxes(),
        self.mujoco_simulation.get_table_dimensions(),
        self.mujoco_simulation.get_placement_area(),
        random_state=random_state,
        max_num_trials=self.mujoco_simulation.max_placement_retry,
        initial_placements= initial_placements
    )
    print('using my version')
    return placement, is_valid



# RearrangeEnv._sample_object_colors = my_sample_object_colors
# RearrangeEnv._sample_random_object_groups = my_sample_random_object_groups

ObjectStateGoal._sample_next_goal_positions = my_sample_next_goal_positions

from robogym.utils.env_utils import load_env
from robogym.utils.parse_arguments import parse_arguments
import matplotlib.pyplot as plt
import pdb
from myrearrange import make_env
from robogym.envs.rearrange.observation.common import MujocoImageObservationProvider


logger = logging.getLogger(__name__)

def create_h5(dataname, cfg, task_params, seqlen, obs_space, act_space):
    H, W, C = obs_space.shape[-3:]
    A = act_space.shape[0]
    num_cameras = eu.get_num_cameras(cfg.env.camera)
    n = cfg.n

    h5 = h5py.File(f'{dataname}.h5', 'w')

    h5.create_dataset('goal', (n, num_cameras, H, W, C))
    h5.create_dataset('image', (n, seqlen, num_cameras, H, W, C))
    h5.create_dataset('action', (n, seqlen, A))  # action space
    h5.create_dataset('reward', (n, seqlen))
    h5.create_dataset('is_first', (n, seqlen))
    h5.create_dataset('is_last', (n, seqlen))
    h5.create_dataset('is_terminal', (n, seqlen))

    h5.attrs['cfg'] = ujson.dumps(cfg.to_dict())
    h5.attrs['task_params'] = ujson.dumps({k: v.tolist() for k, v in task_params.items()})
    h5.attrs['max_reward'] = 0.004  # or is it slightly higher than this? The max I got was 0.00401846
    h5.attrs['action_space'] = act_space.shape
    h5.attrs['observation_space'] = obs_space.shape[-3:]
    return h5


def main():
    """
    Example uses:
        python robogym/scripts/collect_robogym_data.py
    """

    image_size = 256
    # with stabilization, the initial objects got stabilized but the goal objects didn't, 
    # causing an undesired rotation between start and goal state. thus I disabled "stabilize_objects"
    make_env_args = {
        "starting_seed": 2,
        "constants": {"stabilize_objects":False, "normalize_mesh":True, "vision":True, "goal_args": {"randomize_goal_rot": False, "stabilize_goal":False}},
        "parameters": {
            "n_random_initial_steps": 0,
            "simulation_params": {"num_objects": NUM_OBJECTS, 'mesh_scale':1.0, 'used_table_portion': 1.0,
            "camera_fovy_radius": 0.0},
        },
    }
    
    env = make_env(**make_env_args)
    print('env has been made')
    goal_info = env.goal_info()[2]
    assert env is not None, print('doesn\'t seem to be a valid environment')
    obs=env.reset()
    #assert(np.allclose(env.goal_info()[2]['rel_goal_obj_rot'],0,atol=1e-3))
    with env.mujoco_simulation.hide_target(hide_robot=True):
        frame1=env.mujoco_simulation.render(width=image_size,height=image_size,camera_name='vision_cam_top')
    plt.imsave('/share/teststart.png',frame1)
    old_qpos = env.mujoco_simulation.qpos.copy()
    goal_info = env.goal_info()[2]
    goal_qpos = goal_info['goal']['qpos_goal'].copy()
    frames=[]
    with env.mujoco_simulation.hide_target(hide_robot=True):
        for step in range(NUM_OBJECTS):
            env.mujoco_simulation.mj_sim.data.qpos[8+7*step:8+7*(step+1)] = goal_qpos[8+7*step:8+7*(step+1)]
            env.mujoco_simulation.forward()
            frame = env.mujoco_simulation.render(width=image_size,height=image_size,camera_name='vision_cam_top')
            plt.imsave('/share/testframe'+str(step)+'.png',frame)
    vision=env.observe()['vision_goal'][0]
    plt.imsave('/share/testgoal.png',vision)
  
    pdb.set_trace()

    


if __name__ == "__main__":
    # This ensures that we spawn new processes instead of forking. This is necessary
    # rendering has to be in spawned process. GUI cannot be in a forked process.
    # set_start_method('spawn', force=True)
    logging.getLogger("").handlers = []
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    main()


#obs=env.observe()
#obs = env.render(mode="rgb_array", width=256, height=256)

#env, args_remaining = load_env(env_name, return_args_remaining=True, parameters=params)
#config_path = '/home/dayan/Documents/robogym/robogym/scripts/collect_config.jsonnet'
#env = load_env(f"{config_path}::make_env")

#camera_quat:"0.1234811 0.03281631 0.9236646 0.3612743", camera_fovy:"36.392", camera_pos:"0.7477575 0.4143189 -0.5325117"}}#, }
#         max_num_objects: 4,
#         object_size: 0.0254,
#         goal_distance_ratio: 1.0,
#         cast_shadows:: false,

  # goal_img_provider = MujocoGoalImageObservationProvider(
    #                     env.mujoco_simulation,
    #                     'vision_cam_top',
    #                     image_size,
    #                     env.goal_info,
    #                     "qpos_goal",
    #                     hide_robot=True,
    #                 )
    # goal_images = goal_img_provider._render_goal_images(env.goal_info()[2]['goal']['qpos_goal'])
        #ObsProvider = MujocoImageObservationProvider(env.mujoco_simulation, camera_names=['camera_top'], image_size=256)

# constants = dict(normalize_mesh=True, vision=True, vision_args=VisionArgs(
#         image_size=image_size, camera_names=['vision_cam_top']),
#         goal_args=GoalArgs(randomize_goal_rot=False, stabilize_goal=False))
#     params = {'n_random_initial_steps':0,
#         'simulation_params': {
#              'num_objects':num_objects, 'mesh_scale':1.5, 'used_table_portion': 1.0,
#             "camera_fovy_radius": 0.0}}# "camera_pos_radius": 0.00, "camera_quat_radius": 0.0,}}