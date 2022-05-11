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
    MeshRearrangeSimParameters,
)
from robogym.envs.rearrange.goals.object_state_fixed import ObjectFixedStateGoal
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.common.utils import find_meshes_by_dirname, get_combined_mesh
from robogym.observation.common import SyncType
from robogym.envs.rearrange.common.utils import _is_valid_proposal

from data_collection_utils import render_env, unique_deltas, unique_dx, unique_dy, compute_action, get_placement_bounds

import pdb
logger = logging.getLogger(__name__)

 # with stabilization, the initial and goal objects got stabilized differently, 
# causing an undesired rotation between start and goal state. thus I disabled "stabilize_objects"
make_env_args = {
        #"starting_seed": 5,
        "constants": {
            "success_reward":1.0,
            "stabilize_objects":False, "normalize_mesh":True, "vision":True, "vision_args":{
                "image_size":256, 'camera_names':['vision_cam_top']}, "goal_args": {
                    "randomize_goal_rot": False, "stabilize_goal":False}, "success_threshold": 
                    {'obj_pos': 0.1}, "goal_reward_per_object":1.0},
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

    if False:
        # Remove the skinny object meshes
        delete_meshes=[]#'044_flat_screwdriver','042_adjustable_wrench']
        restrict_meshes={}
        for meshname, meshfile in MESH_FILES.items():
            mesh=get_combined_mesh(meshfile)
            # if max(mesh.extents[0]/mesh.extents[1], mesh.extents[1]/mesh.extents[0])>1.7:
            #     delete_meshes.append(meshname)  # delete long skinny objects
            if mesh.extents[2]/max(mesh.extents[0],mesh.extents[1])>1.5:
                delete_meshes.append(meshname)  # delete tall skinny objects:
                #restrict_meshes[meshname] = MESH_FILES[meshname]
        for meshname in delete_meshes:
            del MESH_FILES[meshname]
        #MESH_FILES = restrict_meshes

    def initialize(self):
        super().initialize()
        self.MESH_FILES = find_meshes_by_dirname('ycb')
        self.MESH_FILES.update(find_meshes_by_dirname('geom'))
        num_objects = self.parameters.simulation_params.num_objects
        if num_objects > 4:
            delete_meshes=[]
            for meshname, meshfile in self.MESH_FILES.items():
                mesh=get_combined_mesh(meshfile)
                threshold = 1.4*np.sqrt(2*num_objects)
                normed_extents = mesh.extents * self.constants.normalized_mesh_size/np.max(mesh.extents)
                max_breadth = np.sqrt(normed_extents[0]**2+normed_extents[1]**2)
                if (self.TABLE_WIDTH/(2*max_breadth) < threshold) or (self.TABLE_HEIGHT/(2*max_breadth) < threshold):
                #if (self.TABLE_WIDTH/mesh.extents[0])*(self.TABLE_WIDTH/mesh.extents[1]) < 2*num_objects:
                    delete_meshes.append(meshname)
            for meshname in delete_meshes:
                del self.MESH_FILES[meshname]
            print('choosing from ', len(self.MESH_FILES), ' meshes')
           
    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        return super()._sample_random_object_groups(dedupe_objects=True)

    def _sample_object_colors(self, num_groups):
        all_colors = np.array([[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,0.5,0,1],[1,0,0.5,1],[0.5,0,1,1],[0,0.5,1,1],[0.5,1,0,1],[0,1,0.5,1],[0.25,0,0.5,1]])#,[0.25,0.5,0,1],[0,0.25,0.5,1],[0.5,0.25,0,1],[0.5,0,0.25,1],[0,0.5,0.25,1]])
        # select a random color from the list of colors
        #random_color_ids =  self._random_state.randint(0,len(all_colors),num_groups)
        #random_colors = np.array([all_colors[i] for i in random_color_ids])
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
        goal_image = self.observe()['vision_goal'][0]
        return current_image, goal_image
    
    def i3reset(self):
        _ = self.reset()
        while not self.goal_info()[2]['goal']['goal_valid']:
            print('goal invalid, resetting')
            _  = self.reset()
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
            return _is_valid_proposal(b1_xy[0], b1_xy[1], len(obj_bboxes)-1, obj_bboxes, placements)
        else:
            return False
    
    def pick_and_place(self, action):
        reward = 0 
        obj_positions = self.goal_info()[2]['current_state']['obj_pos'][:,:2]
        success_dist_threshold = self.constants.success_threshold['obj_pos']
        pick_pos = action[:2]
        distances = [np.linalg.norm(obj_pos-pick_pos) for obj_pos in obj_positions]
        closest_obj = np.argmin(distances)
        if distances[closest_obj] < success_dist_threshold:
            if self.is_valid_action(closest_obj, action):
                self.mujoco_simulation.mj_sim.data.qpos[8+7*closest_obj:8+7*closest_obj+2] += action[7:9]
                self.mujoco_simulation.forward()
                self.update_goal_info()
        goal_distances = np.linalg.norm(self.goal_info()[2]['rel_goal_obj_pos'][:,:2],axis=1)
        reward = sum(goal_distances < success_dist_threshold)
        return reward, False

    def i3step(self, action):
        reward, done = self.pick_and_place(action)
        obs = self.i3observe()
        return obs, reward, done, None
    
    def sample_option(self):
        """"
        sample a random action, so a random pick location inside the table placement area
        and one of the 126 possible delta positions
        """
        action = np.zeros(14)
        #action[7:9] = unique_deltas[np.random.choice(len(unique_deltas),size=1)[0],:]
        # delta_x = unique_dx[self._random_state.choice(len(unique_dx),size=1)[0]]
        # delta_y = unique_dy[self._random_state.choice(len(unique_dy),size=1)[0]]
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

    def load_state(self, state):
        pos_data = state[0]
        params = state[1]
        goal = state[2]
        self.constants.normalize_mesh = False
        self._set_object_groups(deepcopy(params.simulation_params.object_groups))
        self._goal = goal
        self._recreate_sim()
        self._apply_object_colors()
        self.sim.set_state(pos_data)
        self.mujoco_simulation.forward()
        self.update_goal_info()
        self._observe_sync(sync_type=SyncType.RESET_GOAL)
        pass

    def load_pickle_state(self, path):
        file_pi = open(path, 'rb') 
        env_state = pickle.load(file_pi)
        self.load_state(env_state)
        file_pi.close()
        pass
      
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

    remove_initial_grid_coords = True
    min_cells = n_objects
    print('num cells : ', n_cells, 'cell width, height:', cell_width, ' ', cell_height )
    if remove_initial_grid_coords:
        min_cells = n_objects*2
    if n_cells < min_cells:
        # Cannot find a valid placement via this method; give up.
        logging.warning(
            f"Unable to fit {n_objects} objects into placement area with {n_cells} cells"
        )
        return np.zeros(shape=(n_objects, 3)), False

    
    # 2. Initialize an array with all valid cell coordinates.
    # Create an array of shape (n_rows, n_columns, 2) where each element contains the row,col
    # coord
    valid_coords = np.dstack(np.mgrid[0:n_rows, 0:n_columns])
    valid_coords = np.reshape(valid_coords, (n_rows * n_columns, 2))
    if len(initial_placements)>0 and remove_initial_grid_coords:
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

        for object_idx in range(n_objects):
            row, col = coords.pop()
            pos, size = object_bounding_boxes[object_idx]

            prop_x = cell_width * col + size[0] - pos[0] 
            prop_y = cell_height * row + size[1] - pos[1]
            #add some jitter
            # jitterx = np.random.uniform(low=-cell_width*0.1*(col==0), high=cell_width*0.1*(col==n_columns-1))
            # jittery = np.random.uniform(low=-cell_height*0.1*(row==0), high=cell_height*0.1*(row==n_rows-1))
            # js=0.2 # cell width and height is around 0.2
            # jitterx = np.random.uniform(low=-cell_width*js, high=cell_width*js)
            # jittery = np.random.uniform(low=-cell_height*js, high=cell_height*js)
            # prop_x = min(cell_width*(n_columns-1) + size[0] - pos[0], max(size[0] - pos[0], prop_x + jitterx))
            # prop_y = min(cell_height*(n_rows-1) + size[1] - pos[1], max(size[1] - pos[1], prop_y + jittery))

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
        placements = placements[len(initial_placements):]
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

make_env = MyRearrangeEnv2.build

if __name__ == "__main__":
    #make_env_args['starting_seed'] = 14
    make_env_args['parameters']["simulation_params"]['num_objects'] = 5
    env = make_env(**make_env_args)
    obs = env.i3reset()
    pdb.set_trace()
    import matplotlib.pyplot as plt
    plt.imsave('/share/testresetStart.png',obs[0])
    plt.imsave('/share/testresetGoal.png',obs[1])
    for idx in range(make_env_args['parameters']["simulation_params"]['num_objects']):
        action = compute_action(env, idx)
        assert(np.allclose(min(np.abs(action[7]-unique_dx)),0,atol=1e-5) and np.allclose(min(abs(action[8]-unique_dy)),0,atol=1e-5))
        assert(env.is_valid_action(idx, action))
        next_obs, reward, done, info = env.i3step(action)
        assert reward == idx+1
        plt.imsave('/share/testAction'+str(idx)+'.png',next_obs[0])
        #assert(action[7:9] in unique_deltas)
        if idx == -2+ make_env_args['parameters']["simulation_params"]['num_objects']:
            env.save_state('/share/test_load_env')
    pdb.set_trace()
    env2 = make_env(**make_env_args)
    env2.load_pickle_state('/share/test_load_env')
    plt.imsave('/share/test_load_env.png', env2.i3observe()[0])
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

