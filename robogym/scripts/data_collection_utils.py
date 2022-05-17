import h5py
import ujson
import numpy as np
from robogym.utils.rotation import euler2quat
import pdb
import matplotlib.pyplot as plt
import pickle

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
    h5.attrs['max_reward'] = 1.0
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
    assert(np.isclose(src_pos+delta_pos, env.goal_info()[2]['goal']['qpos_goal'][8+7*obj_idx:8+7*obj_idx+3],atol=1e-4).all())
    delta_orient = euler2quat(env_info['rel_goal_obj_rot'][obj_idx])
    action = np.concatenate((src_pos, src_orient, delta_pos, delta_orient))
    return action

def render_env(env, config={'image_size':256, 'camera':'vision_cam_top'}):
    with env.mujoco_simulation.hide_target(hide_robot=True):
        return env.mujoco_simulation.render(
            width=config['image_size'],height=config['image_size'],camera_name=config['camera'])/255.0

def get_placement_bounds(env):
    table_pos, table_size, _ = env.mujoco_simulation.get_table_dimensions()
    placement_area = env.mujoco_simulation.get_placement_area()
    offset_x, offset_y, _ = placement_area.offset
    pos_x_min, pos_y_min = [offset_x, offset_y] - table_size[:2] + table_pos[:2]
    width, height, _ = placement_area.size
    pos_x_max, pos_y_max = [pos_x_min, pos_y_min] + np.array([width,height])
    return pos_x_min, pos_x_max, pos_y_min, pos_y_max

def get_unique_position_deltas(dataname):
    f = h5py.File(dataname, 'r')
    all_actions = f['action']
    deltas = f['action'][:,:,7:10].reshape(-1,3)
    starts = f['action'][:,:,:3].reshape(-1,3)
    unique_deltas = np.unique(deltas,axis=0)
    unique_dx = np.unique(deltas[:,0])
    unique_dy = np.unique(deltas[:,1])
    unique_starts = np.unique(starts,axis=0)
    return unique_deltas, unique_dx, unique_dy

def visualize_start_positions(dataname):
    f = h5py.File(dataname, 'r')
    starts = f['action'][:,1,:3].reshape(-1,3)
    fig, ax = plt.subplots()
    hist = ax.hist2d(starts[:,0],starts[:,1], bins=40, density=True)
    fig.colorbar(hist[3], ax=ax)
    xticks = np.arange(min(starts[:,0]), max(starts[:,0]),0.05)
    yticks = np.arange(min(starts[:,1]), max(starts[:,1]),0.05)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.grid(True)
    plt.savefig(dataname + 'start_positions.png')
    pass

def reformat_rewards(dataname):
    f = h5py.File(dataname,'r+')
    num_timesteps = f['reward'].shape[1]
    num_rows = f['reward'].shape[0]
    new_rewards = np.tile(np.arange(num_timesteps),(num_rows,1))
    f['reward'][:] = new_rewards
    f['is_terminal'][:] = np.zeros(f['is_terminal'].shape)
    f.close()
    pass

def merge_h5files(datanames, mergedname):
    pdb.set_trace()
    h5fr = h5py.File(datanames[0],'r')
    cols = list(h5fr.keys())
    all_data={}
    for col in cols:
        all_data[col] = h5fr[col][:]
    for h5name in datanames[1:]:
        h5fr = h5py.File(h5name,'r')
        for col in cols:
            print('stacking column ', col)
            all_data[col] = np.vstack([all_data[col],h5fr[col][:]])
    with h5py.File(mergedname,mode='w') as h5fw:
        for col in cols:
            h5fw[col]=all_data[col]

from robogym.envs.rearrange.common.utils import find_meshes_by_dirname, get_combined_mesh

# def visualize_meshes(meshname_list):
#     all_mesh_files = find_meshes_by_dirname('ycb')
#     all_mesh_files.update(find_meshes_by_dirname('geom'))
#     env = make_env()
#     for meshname in meshname_list:
#         new_mesh_files={}
#         new_mesh_files[meshname] = all_mesh_files[meshname]
#     env.MESH_FILES = new_mesh_files
        

    # with h5py.File(mergedname,mode='w') as h5fw:
    #     row1 = 0
    #     for h5name in datanames:
    #         h5fr = h5py.File(h5name,'r') 
    #         dset1 = list(h5fr.keys())[0]
    #         arr_data = h5fr[dset1][:]
    #         h5fw.require_dataset('alldata', dtype="f",  shape=(50,5), maxshape=(100, 5) )
    #         h5fw['alldata'][row1:row1+arr_data.shape[0],:] = arr_data[:]
    #         row1 += arr_data.shape[0]


#this is for 'data/robogym/20220419212432RGB1.h5'
unique_deltas = np.array([[-0.455625,-0.455625,-0.455625,-0.455625,-0.455625,-0.455625,-0.455625,-0.455625,-0.455625,-0.455625,-0.455625,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.405,-0.3796875,-0.30375,-0.30375,-0.30375,-0.30375,-0.30375,-0.30375,-0.30375,-0.30375,-0.243,-0.2025,-0.2025,-0.2025,-0.2025,-0.2025,-0.2025,-0.2025,-0.2025,-0.2025,-0.2025,-0.151875,-0.151875,-0.151875,-0.151875,-0.151875,-0.151875,-0.151875,-0.151875,-0.151875,-0.151875,-0.1215,-0.1215,-0.1215,-0.1215,-0.10125,-0.08678571,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.1215,0.1215,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.151875,0.2025,0.2025,0.2025,0.2025,0.2025,0.2025,0.2025,0.2025,0.2025,0.2025,0.243,0.243,0.30375,0.30375,0.30375,0.30375,0.30375,0.30375,0.30375,0.30375,0.30375,0.30375,0.30375,0.405,0.405,0.405,0.405,0.405,0.405,0.405,0.405,0.455625,0.455625,0.455625,0.455625,0.455625,0.455625,0.455625,0.5207143],[-0.436335,-0.38785332,-0.29089,-0.19392666,-0.145445,0.,0.145445,0.19392666,0.29089,0.38785332,0.436335,-0.436335,-0.38785332,-0.29089,-0.19392666,-0.145445,-0.116356,0.,0.116356,0.145445,0.19392666,0.29089,0.349068,0.38785332,0.436335,-0.19392666,-0.38785332,-0.29089,-0.19392666,-0.145445,0.,0.145445,0.19392666,0.38785332,0.38785332,-0.436335,-0.38785332,-0.29089,-0.19392666,-0.145445,0.,0.145445,0.19392666,0.29089,0.38785332,-0.38785332,-0.19392666,-0.145445,0.,0.116356,0.145445,0.19392666,0.29089,0.38785332,0.436335,-0.349068,-0.145445,0.,0.19392666,-0.19392666,0.,-0.436335,-0.38785332,-0.29089,-0.19392666,-0.145445,0.145445,0.19392666,0.29089,0.349068,0.38785332,0.436335,0.,0.19392666,-0.465424,-0.436335,-0.38785332,-0.29089,-0.19392666,-0.145445,0.,0.145445,0.19392666,0.29089,0.38785332,0.436335,0.465424,-0.38785332,-0.29089,-0.19392666,-0.145445,0.,0.145445,0.19392666,0.29089,0.38785332,0.436335,-0.19392666,0.,-0.38785332,-0.29089,-0.232712,-0.19392666,0.,0.145445,0.19392666,0.24933429,0.29089,0.38785332,0.436335,-0.38785332,-0.29089,-0.24933429,-0.19392666,0.,0.145445,0.19392666,0.38785332,-0.38785332,-0.19392666,-0.145445,0.,0.19392666,0.29089,0.38785332,-0.19392666]]).T
unique_dx = np.array([-0.455625,-0.405,-0.3796875,-0.30375,-0.243,-0.2025,-0.151875,-0.1215,-0.10125,-0.08678571,0.,0.1215,0.151875,0.2025,0.243,0.30375,0.405,0.455625,0.5207143])
unique_dy = np.array([-0.465424,-0.436335,-0.38785332,-0.349068,-0.29089,-0.24933429,-0.232712,-0.19392666,-0.145445,-0.116356,0.,0.116356,0.145445,0.19392666,0.24933429,0.29089,0.349068,0.38785332,0.436335,0.465424])
x_widths = [0.10125, 0.1215]
if __name__ == "__main__":
    #dataname = '/home/dayan/Documents/docker_share/20220509225239_6objs.h5'
    #dataname = '/home/dayan/Documents/docker_share/20220509225906_8objs.h5'
    #visualize_start_positions('/home/dayan/Documents/implicit-iterative-inference/data/robogym/20220419212432RGB1.h5')
    #visualize_start_positions('/home/dayan/Documents/docker_share/20220509225906_8objs.h5')
    
    dataname='/share/20220419212432RGB1_4objs.h5'
    visualize_start_positions(dataname)
    #print([len(i) for i in get_unique_position_deltas(dataname)])
    # dataname='/share/20220510162221_3objs.h5'
    # reformat_rewards(dataname)
    # f = h5py.File(dataname,'r')
    # print(f['reward'].shape)
    # print(f['reward'][:3])
    # print(f['is_terminal'][:3])
    #dataname='/share/20220419212432RGB1_4objs.h5'
    # d1='/share/20220510173623_3objs.h5'
    # d2='/share/20220510173803_5objs.h5'
    # for dataname in [d1,d2]:
    #     reformat_rewards(dataname)
    #     f = h5py.File(dataname,'r')
    #     print(f['reward'].shape)
    #     print(f['reward'][:3])
    #     print(f['is_terminal'][:3])
    # merge_h5files([d2,'/share/20220510163115_5objs.h5'], '/share/5objs.h5')
    # f = h5py.File('/share/5objs.h5','r')
    # print(f['reward'].shape)
    # plt.imsave('/share/testmergeh5.png',f['image'][0,0,0])
    # pdb.set_trace()