import logging

import numpy as np

config = {'notes':'goal pos jitter', 'n':5000, 'num_objects':4, 'image_size':256, 'camera':'vision_cam_top', 'action_space':14, 'goal_reward':1, }
OUTPUT_DIR = '/share'
DEBUG = False
SAVE_h5 = False
ACTION_SPACE = 14


import matplotlib.pyplot as plt
import pdb
from robogym_wrapper import make_env, make_env_args

from data_collection_utils import create_h5, compute_action, render_env
import datetime
import tqdm
import h5py



logger = logging.getLogger(__name__)

def main():
    """
        python robogym/scripts/collect_robogym_data.py
    """
    # override the default arguments from robogym_wrapper.py
    make_env_args['parameters']["simulation_params"]['num_objects'] = config['num_objects']
    make_env_args['constants']['vision'] = DEBUG # only needed to check env.observe()['vision_goal'][0]
    make_env_args['constants']['success_reward'] = config['goal_reward']
    #make_env_args['starting_seed'] = 15 # 8
    seqlen = config['num_objects'] + 1
    env = make_env(**make_env_args)
    assert env is not None, print('doesn\'t seem to be a valid environment')
    if SAVE_h5:
        dataname = f"{OUTPUT_DIR}/{datetime.datetime.now():%Y%m%d%H%M%S}"
        if DEBUG:
            dataname = dataname + 'debug'
        print(f'Writing to {dataname}...')
        h5 = create_h5(dataname, config, seqlen, make_env_args)
    
        for j in tqdm.tqdm(range(config['n'])):
            obs=env.reset()
            while not env.goal_info()[2]['goal']['goal_valid']:
                print('goal invalid, resetting')
                obs=env.reset()
            assert(env.goal_info()[2]['goal']['goal_valid'])
            assert(np.allclose(env.goal_info()[2]['rel_goal_obj_rot'],0,atol=1e-3))
            action = np.zeros(ACTION_SPACE)
            reward, done = 0, False
            goal_qpos = env.goal_info()[2]['goal']['qpos_goal'].copy()
            h5['image'][j, 0] = render_env(env, config)
            h5['action'][j, 0] = action
            h5['reward'][j, 0] = reward
            h5['is_first'][j, 0] = True
            h5['is_last'][j, 0] = False
            h5['is_terminal'][j, 0] = done

            # Place each object one by one:
            for idx in range(config['num_objects']):
                t = idx+1
                is_last = (t >= seqlen-1)
                action = compute_action(env, idx)
                env.mujoco_simulation.mj_sim.data.qpos[8+7*idx:8+7*(idx+1)] = goal_qpos[8+7*idx:8+7*(idx+1)]
                env.mujoco_simulation.forward()
                env.update_goal_info()
                h5['image'][j, t] = render_env(env, config)
                h5['action'][j, t] = action
                h5['reward'][j, t] = config['goal_reward']*is_last
                h5['is_first'][j, t] = False
                h5['is_last'][j, t] = is_last
                h5['is_terminal'][j, t] = is_last
            h5['goal'][j] = h5['image'][j, t].copy()
    
    if DEBUG:
        if not SAVE_h5:
            obs=env.reset()
            while not env.goal_info()[2]['goal']['goal_valid']:
                print('goal invalid, resetting')
                obs=env.reset()
            old_qpos = env.mujoco_simulation.qpos.copy()
        env.mujoco_simulation.mj_sim.data.qpos[:] = old_qpos
        env.mujoco_simulation.forward()
        with env.mujoco_simulation.hide_target(hide_robot=True):
            frame1=env.mujoco_simulation.render(width=config['image_size'],height=config['image_size'],camera_name=config['camera'])
        plt.imsave('/share/teststart.png',frame1)
       
        #env.parameters.simulation_params.object_groups[3].mesh_files
        goal_qpos = env.goal_info()[2]['goal']['qpos_goal'].copy()
        if SAVE_h5:
            f = h5py.File(dataname +'.h5', 'r')
            plt.imsave('/share/testh5.png',f['image'][-1,0,0])
        with env.mujoco_simulation.hide_target(hide_robot=True):
            for obj in range(config['num_objects']):
                env.mujoco_simulation.mj_sim.data.qpos[8+7*obj:8+7*(obj+1)] = goal_qpos[8+7*obj:8+7*(obj+1)]
                env.mujoco_simulation.forward()
                frame = env.mujoco_simulation.render(width=config['image_size'],height=config['image_size'],camera_name=config['camera'])
                plt.imsave('/share/testframe'+str(obj)+'.png',frame)
        pdb.set_trace()
        vision=env.observe()['vision_goal'][0]
        plt.imsave('/share/testgoal.png',vision)
        

if __name__ == "__main__":
    # This ensures that we spawn new processes instead of forking. This is necessary
    # rendering has to be in spawned process. GUI cannot be in a forked process.
    # set_start_method('spawn', force=True)
    logging.getLogger("").handlers = []
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    main()

#    old_qpos = env.mujoco_simulation.qpos.copy()

# RearrangeEnv._sample_object_colors = my_sample_object_colors
# RearrangeEnv._sample_random_object_groups = my_sample_random_object_groups

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
    #                     IMAGE_SIZE,
    #                     env.goal_info,
    #                     "qpos_goal",
    #                     hide_robot=True,
    #                 )
    # goal_images = goal_img_provider._render_goal_images(env.goal_info()[2]['goal']['qpos_goal'])
        #ObsProvider = MujocoImageObservationProvider(env.mujoco_simulation, camera_names=['camera_top'], IMAGE_SIZE=256)

# constants = dict(normalize_mesh=True, vision=True, vision_args=VisionArgs(
#         IMAGE_SIZE=IMAGE_SIZE, camera_names=['vision_cam_top']),
#         goal_args=GoalArgs(randomize_goal_rot=False, stabilize_goal=False))
#     params = {'n_random_initial_steps':0,
#         'simulation_params': {
#              'num_objects':num_objects, 'mesh_scale':1.5, 'used_table_portion': 1.0,
#             "camera_fovy_radius": 0.0}}# "camera_pos_radius": 0.00, "camera_quat_radius": 0.0,}}