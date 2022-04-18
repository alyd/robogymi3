from robogym.envs.rearrange.common.base import RearrangeEnvConstants
from robogym.envs.rearrange.goals.pickandplace import PickAndPlaceGoal
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.ycb import YcbRearrangeEnv
import numpy as np


class MyRearrangeEnv(YcbRearrangeEnv):

    def _sample_object_colors(self, num_groups):
        all_colors = [[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,0.5,0,1],[1,0,0.5,1],[0.5,0,1,1],[0,0.5,1,1],[0.5,1,0,1],[0,1,0.5,1],[0.25,0,0.5,1]]
        # select a random color from the list of colors
        random_color_ids =  self._random_state.randint(0,len(all_colors),num_groups)
        random_colors = np.array([all_colors[i] for i in random_color_ids])
        return random_colors

    def _sample_random_object_groups(
        self, dedupe_objects):
        return super()._sample_random_object_groups(dedupe_objects=True)


make_env = MyRearrangeEnv.build
