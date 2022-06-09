from functools import partial
import logging
from typing import List

import attr
import numpy as np
import pickle
from copy import deepcopy

from robogym.envs.rearrange.common.mesh import (
    MeshRearrangeEnv,
    MeshRearrangeEnvConstants,
    MeshRearrangeEnvParameters,
)
from robogym.envs.rearrange.common.utils import geom_ids_of_body
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.common.utils import find_meshes_by_dirname, get_combined_mesh
from robogym.mujoco.mujoco_xml import ASSETS_DIR
from robogym.observation.common import SyncType
from robogym.envs.rearrange.common.utils import _is_valid_proposal

from data_collection_utils import render_env, unique_deltas, unique_dx, unique_dy, compute_action, get_placement_bounds
import matplotlib.pyplot as plt
import pdb
logger = logging.getLogger(__name__)

VIZ_MESH_NAMES = None#['009_gelatin_box','050_medium_clamp'] #['002_master_chef_can', '073-f_lego_duplo']
 # with stabilization, the initial and goal objects got stabilized differently, 
# causing an undesired rotation between start and goal state. thus I disabled "stabilize_objects"
MAKE_ENV_ARGS = {
        #"starting_seed": 5,
        "constants": {
            "success_reward":1.0,
            "stabilize_objects":False, "normalize_mesh":True, "vision":True, "vision_args":{
                "image_size":256, 'camera_names':['vision_cam_top']}, "goal_args": {
                    "randomize_goal_rot": False, "stabilize_goal":False}, "success_threshold": 
                    {'obj_pos': 0.05}, "goal_reward_per_object":1.0},
        "parameters": {
            "n_random_initial_steps": 0,
            "simulation_params": {"num_objects": 4, 'mesh_scale':1.5, 'used_table_portion': 1.0,
            "camera_fovy_radius": 0.0},
        },
    }

class MyRearrangeEnv2(
    MeshRearrangeEnv[
        MeshRearrangeEnvParameters, MeshRearrangeEnvConstants, MeshRearrangeSim,
    ]
):
    
    TABLE_WIDTH = 0.6075
    TABLE_HEIGHT = 0.58178

    def initialize(self):
        super().initialize()
        self.task = lambda: None
        self.make_args = MAKE_ENV_ARGS
        self.num_objects = self.parameters.simulation_params.num_objects
        self.make_args['parameters']["simulation_params"]['num_objects'] = self.num_objects
        self.num_constraints_to_satisfy = self.num_objects #TODO: partial goals
        if self.num_objects < 6:
            setattr(self.task,'max_moves_required', self.num_objects)
            self.num_constraints_already_satisfied = 0
            self.num_additional_constraints_to_satisfy = self.num_objects
        else:
            setattr(self.task,'max_moves_required', 4)
            self.num_constraints_already_satisfied = self.num_objects - 4
            self.num_additional_constraints_to_satisfy = 4
        self.MESH_FILES = find_meshes_by_dirname('ycb')
        self.MESH_FILES.update(find_meshes_by_dirname('geom'))
        print('choosing from ', len(self.MESH_FILES), ' meshes')
           
    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        return super()._sample_random_object_groups(dedupe_objects=True)

    def _sample_object_colors(self, num_groups):
        all_colors = np.array([[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,0.5,0,1],[1,0,0.5,1],[0.5,0,1,1],[0,0.5,1,1],[0.5,1,0,1],[0,1,0.5,1],[0.25,0,0.5,1]])#,[0.25,0.5,0,1],[0,0.25,0.5,1],[0.5,0.25,0,1],[0.5,0,0.25,1],[0,0.5,0.25,1]])
        # select a random color from the list of colors
        random_color_ids =  self._random_state.choice(len(all_colors), num_groups, replace=False)
        random_colors = all_colors[random_color_ids]
        return random_colors

    def _sample_object_meshes(self, num_groups: int) -> List[List[str]]:
        candidates = list(self.MESH_FILES.values())
        assert len(candidates) > 0, f"No mesh file for {self.parameters.mesh_names}."
        candidates = sorted(candidates)
        indices = self._random_state.choice(
            len(candidates), size=num_groups, replace=True
        )
        return [candidates[i] for i in indices]

    def reset_goal(
        self, update_seed=False, sync_type=SyncType.RESET_GOAL
    ):
        """
        Modified from the original so that it won't raise an error of the goal is invalid.
        Our code will check and automatically regenerate the goal if it's invalid
        """
        obs = super().reset_goal(update_seed=update_seed, sync_type=sync_type, raise_when_invalid=False)
        # Check if goal placement is valid here.
        if not self._goal["goal_valid"]:
            logger.info(
                "InvalidSimulationError: "
                + self._goal.get("goal_invalid_reason", "Goal is invalid.")
            )
        return obs

    def _randomize_object_initial_positions(self):
        """
        Randomize initial position for each object.
        """
        object_pos, is_valid = self._generate_object_placements()

        while not is_valid:
            print("Object initial placement is invalid, regenerating")
            object_pos, is_valid = self._generate_object_placements()

        self.mujoco_simulation.set_object_pos(object_pos)

    # The i3 methods handle actions and return observations of the type used
    # in implicit-iterative-inference
    def i3observe(self):
        current_image = render_env(self)
        return current_image, self.goal_image

    def handle_partial_constraints(self, partial_constraints):
        self.partial_constraints = partial_constraints
        if self.partial_constraints ==-1:
            self.goal_image = self.observe()['vision_goal'][0]/255.0
        else:
            self.num_constraints_to_satisfy = partial_constraints
            setattr(self.task,'max_moves_required', partial_constraints)
            self.num_constraints_already_satisfied = 0
            self.num_additional_constraints_to_satisfy = partial_constraints
            stata = self.get_state_data()
            make_args = self.make_args
            make_args['parameters']["simulation_params"]['num_objects'] = self.num_objects
            goal_env = make_env(**make_args)
            goal_env.load_state(stata)
            goal_qpos = self.goal_info()[2]['goal']['qpos_goal'].copy()
            for idx in range(self.partial_constraints):
                goal_env.mujoco_simulation.mj_sim.data.qpos[8+7*idx:8+7*(idx+1)] = goal_qpos[8+7*idx:8+7*(idx+1)]
                goal_env.mujoco_simulation.forward()
            geom_ids_to_hide = [
                target_id
                for i in range(self.partial_constraints, self.num_objects)
                for target_id in geom_ids_of_body(goal_env.mujoco_simulation.mj_sim, f"object{i}")
            ]
            goal_env.mujoco_simulation.mj_sim.model.geom_rgba[geom_ids_to_hide, -1] = 0.0
            self.goal_image = render_env(goal_env)
        obs = self.i3observe() # needs to be called an extra time to refresh?
    
    def i3reset(self, partial_constraints=-1):
        _ = self.reset()
        while not self.goal_info()[2]['goal']['goal_valid']:
            print('goal invalid, resetting')
            _  = self.reset()
        self.handle_partial_constraints(partial_constraints)
        return self.i3observe()

    def is_valid_action(self, object_idx, action):
        obj_bboxes = np.copy(self.mujoco_simulation.get_object_bounding_boxes())
        mesh_pos, mesh_size = obj_bboxes[object_idx]
        placements = np.copy(self.goal_info()[2]['current_state']['obj_pos'][:,:3])
        b1_xy = placements[object_idx][:2] + action[7:9]
        # is the object going to still be on the table?
        pos_x_min, pos_x_max, pos_y_min, pos_y_max = get_placement_bounds(self)
        mins = [pos_x_min, pos_y_min]+mesh_size[:2]-mesh_pos[:2]
        maxes = [pos_x_max, pos_y_max] - mesh_size[:2] - mesh_pos[:2]
        above_min = np.logical_or(np.greater(b1_xy, mins),np.isclose(b1_xy, mins, atol=1e-5))
        below_max = np.logical_or(np.less(b1_xy, maxes),np.isclose(b1_xy, maxes, atol=1e-5))
        if above_min.all() and below_max.all(): # check for collisions
            placements = np.delete(placements,object_idx, axis=0)
            obj_bboxes = np.vstack([np.delete(obj_bboxes,object_idx,axis=0),obj_bboxes[object_idx:object_idx+1]])
            if _is_valid_proposal(b1_xy[0], b1_xy[1], len(obj_bboxes)-1, obj_bboxes, placements):
                return True
            else:
                # print('invalid action, the object would collide with another object')
                return False
        else:
            # print('invalid action, the object would be off the table')
            return False
    
    def pick_and_place(self, action, action_noise_scale=0):
        done = False 
        obj_positions = self.goal_info()[2]['current_state']['obj_pos'][:,:2]
        success_dist_threshold = self.constants.success_threshold['obj_pos']
        if action_noise_scale > 0:
            noise_vec = self._random_state.normal(loc=0,scale=action_noise_scale,size=len(action))
            action += noise_vec
        pick_pos, delta_pos = action[:2], action[7:9]
        distances = [np.linalg.norm(obj_pos-pick_pos) for obj_pos in obj_positions]
        closest_obj = np.argmin(distances)
        #print('pick position: ', pick_pos, "delta position: ", delta_pos)
        # print("closest object idx: ", closest_obj, " distance: ", distances[closest_obj])
        if distances[closest_obj] < success_dist_threshold:
            if self.is_valid_action(closest_obj, action):
                self.mujoco_simulation.mj_sim.data.qpos[8+7*closest_obj:8+7*closest_obj+2] += delta_pos
                self.mujoco_simulation.forward()
                self.update_goal_info()

        goal_distances = np.linalg.norm(self.goal_info()[2]['rel_goal_obj_pos'][:,:2],axis=1)
        if self.partial_constraints > 0:
            goal_distances = goal_distances[:self.partial_constraints]
        if max(goal_distances) < success_dist_threshold:
            done = True
        reward = sum(goal_distances < success_dist_threshold)
        return reward, done

    def null_action(self):
        action = self.sample_option()
        action = action*0
        return action 

    def i3step(self, action, action_noise_scale=0):
        reward, done = self.pick_and_place(action, action_noise_scale=action_noise_scale)
        obs = self.i3observe()
        return obs, reward, done, None
    
    def sample_option(self):
        """"
        sample a random action, so a random pick location inside the table placement area
        and a delta position that will keep the object within the placement area
        """
        action = np.zeros(14)
        pos_x_min, pos_x_max, pos_y_min, pos_y_max = get_placement_bounds(self)
        pos_x = self._random_state.uniform(low=pos_x_min,high=pos_x_max)
        pos_y = self._random_state.uniform(low=pos_y_min,high=pos_y_max)
        delta_x = self._random_state.uniform(low=pos_x_min - pos_x, high=pos_x_max - pos_x)
        delta_y = self._random_state.uniform(low=pos_y_min - pos_y, high=pos_y_max - pos_y)
        action[7:9] = delta_x, delta_y
        action[:2] = pos_x, pos_y
        return action

    def get_state_data(self):
        pos_state = self.sim.get_state()
        return deepcopy((pos_state,self.parameters,self._goal))

    def save_state(self, path):
        file_pi = open(path, 'wb')
        pickle.dump(self.get_state_data(), file_pi)
        file_pi.close()
        pass

    def load_state(self, state, partial_constraints=-1):
        pos_data = state[0]
        params = state[1]
        goal = state[2]
        self.constants.normalize_mesh = False
        obj_groups = deepcopy(params.simulation_params.object_groups)
        for group in obj_groups:
            group.mesh_files = [ASSETS_DIR + mesh_file.split('assets')[1] for mesh_file in group.mesh_files]
        self._set_object_groups(obj_groups)
        self._goal = goal
        self._recreate_sim()
        self._apply_object_colors()
        self.sim.set_state(pos_data)
        self.mujoco_simulation.forward()
        self.update_goal_info()
        self._observe_sync(sync_type=SyncType.RESET_GOAL)
        self.handle_partial_constraints(partial_constraints)
        pass

    def load_pickle_state(self, path, partial_constraints=-1):
        file_pi = open(path, 'rb') 
        env_state = pickle.load(file_pi)
        self.load_state(env_state, partial_constraints=partial_constraints)
        file_pi.close()
        pass

make_env = MyRearrangeEnv2.build



# WIP:
# def make_env(make_env_args, max_moves_required):
#     env = MyRearrangeEnv2.build(**make_env_args)
#     num_unchanged_objs = make_env_args['parameters']["simulation_params"]['num_objects'] - max_moves_required
#     for obj_id in range(num_unchanged_objs):
#         action = compute_action(env, obj_id)
#         _ = env.pick_and_place(action)
#     goal_distances = np.linalg.norm(self.goal_info()[2]['rel_goal_obj_pos'][:,:2],axis=1)
#     assert(sum(goal_distances < 0.001)==num_unchanged_objs)
#     return env
        
      
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal

def my_place_objects_in_grid(
    object_bounding_boxes,
    table_dimensions,
    placement_area,
    random_state,
    max_num_trials = 5, initial_placements=[]):
    """
    Modified from the original to avoid placing any goal objects overlapping with any initial objects

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

    def _get_local_coords(global_placement, object_bounding_box, cell_width, cell_height):
        pos, size = object_bounding_box
        prop = global_placement - [offset_x, offset_y, 0.0] + table_size - table_pos
        col = (prop[0] + pos[0] - size[0])/cell_width
        row = (prop[1] + pos[1] - size[1])/cell_height
        assert np.allclose(round(col), col, atol=1e-6) and np.allclose(round(row), row, atol=1e-6)
        return round(row), round(col)

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

    print('num cells : ', n_cells, 'cell width, height:', cell_width, ' ', cell_height )
    
    if n_objects < 6 or len(initial_placements)==0:
        n_goals_to_place = n_objects
    else: # with more than 5 objects, we will only have 4 goal objects in a different location
        n_goals_to_place = 4
    if n_cells < n_objects + n_goals_to_place:
        # Cannot find a valid placement via this method; give up.
        logging.warning(
            f"Can't fit {n_objects} initial objects & {n_goals_to_place} extra goal objects into placement area with {n_cells} cells"
        )
        return np.zeros(shape=(n_objects, 3)), False
    # 2. Initialize an array with all valid cell coordinates.
    # Create an array of shape (n_rows, n_columns, 2) where each element contains the row,col
    # coord
    valid_coords = np.dstack(np.mgrid[0:n_rows, 0:n_columns])
    valid_coords = np.reshape(valid_coords, (n_rows * n_columns, 2))
    if len(initial_placements)>0:
        initial_coords = [_get_local_coords(initial_placements[idx], object_bounding_boxes[idx], cell_width, cell_height) for idx in range(n_objects)]
        valid_coords = np.delete(valid_coords, [n_columns*ic[0]+ic[1] for ic in initial_coords], 0)
    for trial_i in range(max_num_trials):
        placement_valid = True
        placements = [i for i in initial_placements]
        # 2. Initialize an array with all valid cell coordinates.
        coords = np.copy(valid_coords)
        # Create a shuffled list where ever entry is a valid (row, column) coordinate.
        random_state.shuffle(coords)
        coords = list(coords)

        # 3. Place each object into a randomly selected cell.
        object_bounding_boxes_including_initial = np.vstack([object_bounding_boxes, object_bounding_boxes])

        for object_idx in range(n_goals_to_place):
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
                if (trial_i+1) % 30 == 0:
                    logging.warning(f"Trial {trial_i} failed on object {object_idx}")
                break

            placements.append(placement)
        # the objects with unsatisfied goals are at the beginning of the list
        placements = placements[len(initial_placements):] + placements[n_goals_to_place:n_objects]
        if placement_valid:
            assert (
                len(placements) == n_objects
            ), "There should be a placement for every object"
            break

    return np.array(placements), placement_valid

def my_sample_next_goal_positions(self, random_state):
    """
    Modified from the original to call my_place_objects_in_grid with the initial object locations,
    which ensures goal objects won't overlap with initial objects
    """
    initial_placements = []
    num_objects = int((self.mujoco_simulation.qpos.size-8)/7)
    for step in range(num_objects):
        if np.all(self.mujoco_simulation.qpos[8+7*step:8+7*step+3]!=0): # qpos=0 when make_env is called, we only need to do this after env.reset()
            initial_placements.append(self.mujoco_simulation.qpos[8+7*step:8+7*step+3])
    placement, is_valid = my_place_objects_in_grid(
        self.mujoco_simulation.get_object_bounding_boxes(),
        self.mujoco_simulation.get_table_dimensions(),
        self.mujoco_simulation.get_placement_area(),
        random_state=random_state,
        max_num_trials=self.mujoco_simulation.max_placement_retry,
        initial_placements= initial_placements,
    )
    return placement, is_valid

ObjectStateGoal._sample_next_goal_positions = my_sample_next_goal_positions


if __name__ == "__main__":
    MAKE_ENV_ARGS['starting_seed'] =7
    MAKE_ENV_ARGS['parameters']["simulation_params"]['num_objects'] = 4
    env = make_env(**MAKE_ENV_ARGS)
    # with open('/home/dayan/Documents/docker_share/env_states20220519151709', 'rb') as file_pi:
    #     env_dict = pickle.load(file_pi)
    # tasks = env_dict[5]
    # env.load_state(tasks[3], partial_constraints=-1)
    obs = env.i3reset()
    action = compute_action(env, 0)
    print(action)
    print(env.null_action())
    pdb.set_trace()
    plt.imsave('visualizations/frame0.png',obs[0])
    goal_distances = np.linalg.norm(env.goal_info()[2]['rel_goal_obj_pos'][:,:2],axis=1)
    print(goal_distances)
    for idx in range(env.parameters.simulation_params.num_objects):
        action = compute_action(env, idx)
        next_obs, reward, done, info = env.i3step(action)
        print(reward)
        goal_distances = np.linalg.norm(env.goal_info()[2]['rel_goal_obj_pos'][:,:2],axis=1)
        print(goal_distances)
        pdb.set_trace()
        plt.imsave('visualizations/frame'+str(idx+1)+'.png',next_obs[0])
    pdb.set_trace()
    obs = env.i3observe()
    plt.imsave('test_load_env.png', obs[0])
    plt.imsave('test_load_env2.png', obs[1])
    pdb.set_trace()
    # env.save_state('test_save_partial')
    obs = env.i3reset(partial_constraints=2)
    plt.imsave('testpartialStart.png',obs[0])
    plt.imsave('testpartialGoal.png',obs[1])
    pdb.set_trace()
    env2 = make_env(**MAKE_ENV_ARGS)
    env2.load_pickle_state('test_save_partial', partial_constraints=2)
    obs = env2.i3observe()
    plt.imsave('test_load_env.png', obs[0])
    plt.imsave('test_load_env2.png', obs[1])
    pdb.set_trace()
    #env.save_state('/share/test_load_env')
    # plt.imsave('/home/dayan/Documents/docker_share/testresetStart.png',obs[0])
    # if VIZ_MESH_NAMES is not None:
    #     plt.imsave('/home/dayan/Documents/docker_share/meshes/'+str(VIZ_MESH_NAMES)+'.png',obs[0])
    # plt.imsave('/home/dayan/Documents/docker_share/testresetGoal.png',obs[1])
    # pdb.set_trace()
    for idx in range(MAKE_ENV_ARGS['parameters']["simulation_params"]['num_objects']):
        action = compute_action(env, idx)
        assert(env.is_valid_action(idx, action))
        next_obs, reward, done, info = env.i3step(action)
        if idx < env.partial_constraints:
            assert reward == idx+1
        else:
            assert reward == env.partial_constraints
        plt.imsave('testAction'+str(idx)+'.png',next_obs[0])
        print(reward)
    #     if idx == -2+ make_env_args['parameters']["simulation_params"]['num_objects']:
    #         env.save_state('/share/test_load_env')
    pdb.set_trace()
    MAKE_ENV_ARGS['parameters']["simulation_params"]['num_objects'] = 6
    env2 = make_env(**MAKE_ENV_ARGS)
    pdb.set_trace()
    env2.load_pickle_state('/home/dayan/Documents/docker_share/test_load_env')
    obs = env2.i3observe()
    plt.imsave('/home/dayan/Documents/docker_share/test_load_env.png', env2.i3observe()[0])
    test_action = compute_action(env2, idx)
    assert np.allclose(test_action, action)
    test_obs, reward, done, info = env2.i3step(action)
    assert np.allclose(test_obs[0], next_obs[0])
    assert reward == idx+1
    pdb.set_trace()
    print(env.sample_option())
    invalid_actions = [np.array([1.22, 0.84, 0.49, 0,0,0,0, 0,0.1,0,0,0,0,0]),np.array([1.22, 0.84, 0.49, 0,0,0,0, 0.2,0,0,0,0,0,0])]
    # action would bring object off the table, action would collide with another object
    for action in invalid_actions:
        assert not env.is_valid_action(0,action)
        # check action isn't taken:
        invalid_obs, reward, done, info = env.i3step(np.array([1.22, 0.84, 0.49, 0,0,0,0, 0,0.1,0,0,0,0,0]))
        assert((invalid_obs[0]==next_obs[0]).all())
    pdb.set_trace()
    plt.imsave('/share/testStepStart.png',next_obs[0])
    plt.imsave('/share/testStepGoal.png',next_obs[1])
    rel_goal_pos = env.goal_info()[2]['rel_goal_obj_pos']
    current_pos = env.goal_info()[2]['current_state']['obj_pos']
    pdb.set_trace()

