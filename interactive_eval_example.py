import sys
sys.path.append("robogym/scripts")
import robogym_wrapper as rw
import pickle
import utils

NUM_OBJECTS = 4
NUM_TASKS = 10
HORIZON = 8
robo_env_args = rw.make_env_args
robo_env_args['starting_seed'] = 0
robo_env_args['parameters']["simulation_params"]['num_objects'] = NUM_OBJECTS
# determines when an object is close enough to the goal to count as a match,
# or a pick location is close enough to the object to successfully pick it up:
robo_env_args['constants']["success_threshold"]['obj_pos'] = 0.05 
env = rw.make_env(**robo_env_args)

# it is possible to generate random robogym tasks from scratch, but with more objects
# the process can take 5-15 seconds each time. to save time, load presaved tasks

premade_tasks_path = '/home/dayan/Documents/docker_share/env_states20220513172222'
if premade_tasks_path is not None:
    with open(premade_tasks_path, 'rb') as file_pi:
        env_dict = pickle.load(file_pi)
    tasks = env_dict[NUM_OBJECTS] # this is a list of 200 pregenerated tasks with NUM_OBJECTS objects

for i in range(NUM_TASKS):
    # i3reset and i3observe generate image observations in the same format as the block envs 
    if premade_tasks_path is not None:
        env.load_state(tasks[i]) # loads start and goal object meshes, scales, colors, positions, and orientations
        obs, done = env.i3observe(), False
    else:
        obs, done = env.i3reset(), False
    for t in range(HORIZON):
        action = model(obs)
        # use i3step() instead of step() to handle actions in the same format as the block envs
        next_obs, reward, done, info = env.i3step(utils.to_np(action))