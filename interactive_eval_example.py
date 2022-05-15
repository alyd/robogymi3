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
premade_tasks_path = '/home/dayan/Documents/docker_share/env_states20220513172222'
if premade_tasks_path is not None:
    with open(premade_tasks_path, 'rb') as file_pi:
        env_dict = pickle.load(file_pi)
    tasks = env_dict[NUM_OBJECTS] # this is a list of 200 pregenerated environment states with NUM_OBJECTS objects

for i in range(NUM_TASKS):
    if premade_tasks_path is not None:
        env.load_state(tasks[i])
        obs, done = env.i3observe(), False
    else:
        obs, done = env.i3reset(), False
    for t in range(HORIZON):
        # use i3step() instead of step() to handle actions 
        next_obs, reward, done, info = env.i3step(utils.to_np(action))