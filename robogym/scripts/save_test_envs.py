import logging

import numpy as np

config = {'object_nums':[3,4,5,6,7,8], 'notes':'all 94 meshes, 6-8 have 4 unsatisfied goals', 'scale':1.5, 'n':200, 'image_size':256, 'camera':'vision_cam_top', 'action_space':14,}
OUTPUT_DIR = '/share'
DEBUG = False

import matplotlib.pyplot as plt
import pdb
from robogym_wrapper import make_env, make_env_args
import pickle

from data_collection_utils import create_h5, compute_action, render_env
import datetime
import tqdm
from copy import deepcopy


logger = logging.getLogger(__name__)

def main():
    """
        python robogym/scripts/save_test_envs.py
        /share/env_states20220509194443 has 3,5,6,8
        /share/env_states20220510100023 has 100 states, 4 objects of all meshes, scale 1.5
        /share/env_states20220517161902 has 500 states, 4 objects of all meshes, scale 1.5
        /share/env_states20220518141617 has 200 states, 6 objects "" ""
        /share/env_states20220519151709 has 200 states each, 3,4,5,6,7,8 objects of all objects, scale 1.5, 6-8 have 4 goals unsatisfied
        /share/env_states..2222 has 3,4,5,6 objects, 55 meshes
        env_states_scale19_bigmeshes has 4 objects, 60 meshes (no skinny or small, scale 1.9)
        
    """
    # override the default arguments from robogym_wrapper.py
    #make_env_args['starting_seed'] = 15 # 8
    dataname = f"{OUTPUT_DIR}/env_states{datetime.datetime.now():%Y%m%d%H%M%S}"
    if DEBUG:
        dataname = dataname + 'debug'
    print(f'Writing to {dataname}...')
    states = {'scale':config['scale'], 'notes': config['notes']}
    
    for object_num in config['object_nums']:
        states[object_num] = []
        make_env_args['parameters']["simulation_params"]['num_objects'] = object_num
        env = make_env(**make_env_args)
        assert env is not None, print('doesn\'t seem to be a valid environment')
        for j in tqdm.tqdm(range(config['n'])):
            obs=env.reset()
            while not env.goal_info()[2]['goal']['goal_valid']:
                print('goal invalid, resetting')
                obs=env.reset()
            assert(env.goal_info()[2]['goal']['goal_valid'])
            assert(np.allclose(env.goal_info()[2]['rel_goal_obj_rot'],0,atol=1e-3))
            states[object_num].append(env.get_state_data())

    file_pi = open(dataname, 'wb') 
    pickle.dump(states, file_pi)
    file_pi.close()
    print('saved to ', dataname)

    if DEBUG:
        file_pi = open(dataname, 'rb') 
        states = pickle.load(file_pi)
        file_pi.close()
        make_env_args['parameters']["simulation_params"]['num_objects'] = 5
        env2 = make_env(**make_env_args)
        env2.load_state(states[5][0])
        plt.imsave('/share/test_load_env.png', env2.i3observe()[0])
        env2 = make_env(**make_env_args)
        env2.load_state(states[5][1])
        plt.imsave('/share/test_load_env2.png', env2.i3observe()[0])
        action = compute_action(env2, 0)
        next_obs, reward, done, info = env2.i3step(action)
        pdb.set_trace()
        assert reward == 1
        plt.imsave('/share/test_load_envAction.png',next_obs[0])
        plt.imsave('/share/test_load_envActiongoal.png', next_obs[1])
        pdb.set_trace()

if __name__ == "__main__":
    # This ensures that we spawn new processes instead of forking. This is necessary
    # rendering has to be in spawned process. GUI cannot be in a forked process.
    # set_start_method('spawn', force=True)
    logging.getLogger("").handlers = []
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    main()