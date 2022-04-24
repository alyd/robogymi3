import h5py
import ujson
import numpy as np
from robogym.utils.rotation import euler2quat
import pdb

def create_h5(dataname, cfg, seqlen, make_env_args):
    H, W, C = cfg['image_size'], cfg['image_size'], 3
    A = cfg['action_space']
    num_cameras = 1
    n = cfg['n']

    h5 = h5py.File(f'{dataname}.h5', 'w')

    h5.create_dataset('goal', (n, num_cameras, H, W, C))
    h5.create_dataset('image', (n, seqlen, num_cameras, H, W, C))
    h5.create_dataset('action', (n, seqlen, A))  # action space
    h5.create_dataset('reward', (n, seqlen))
    h5.create_dataset('is_first', (n, seqlen))
    h5.create_dataset('is_last', (n, seqlen))
    h5.create_dataset('is_terminal', (n, seqlen))

    h5.attrs['cfg'] = ujson.dumps(cfg)
    h5.attrs['make_env_args'] = ujson.dumps(make_env_args)
    h5.attrs['max_reward'] = 0.004  # or is it slightly higher than this? The max I got was 0.00401846
    # h5.attrs['action_space'] = act_space.shape
    # h5.attrs['observation_space'] = obs_space.shape[-3:]
    return h5


def compute_action(env, obj_idx):
    env_info = env.goal_info()[2]
    # get current object state
    current_state = env_info['current_state']
    src_pos = current_state['obj_pos'][obj_idx]
    assert((src_pos==env.mujoco_simulation.mj_sim.data.qpos[8+7*obj_idx:8+7*obj_idx+3]).all())
    src_orient = euler2quat(current_state['obj_rot'][obj_idx])
    # get target object state
    delta_pos = env_info['rel_goal_obj_pos'][obj_idx]
    assert(((src_pos+delta_pos)==env.goal_info()[2]['goal']['qpos_goal'][8+7*obj_idx:8+7*obj_idx+3]).all())
    delta_orient = euler2quat(env_info['rel_goal_obj_rot'][obj_idx])
    action = np.concatenate((src_pos, src_orient, delta_pos, delta_orient))
    return action

def render_env(env, config={'image_size':256, 'camera':'vision_cam_top'}):
    with env.mujoco_simulation.hide_target(hide_robot=True):
        return env.mujoco_simulation.render(
            width=config['image_size'],height=config['image_size'],camera_name=config['camera'])/255.0