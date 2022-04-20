import logging
from typing import List

import attr
import numpy as np

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
from robogym.robot_env import build_nested_attr
from robogym.utils.rotation import quat_from_angle_and_axis
from robogym.observation.common import SyncType

import pdb
logger = logging.getLogger(__name__)


class MyRearrangeEnv2(
    MeshRearrangeEnv[
        MeshRearrangeEnvParameters, MeshRearrangeEnvConstants, MeshRearrangeSim,
    ]
):
    MESH_FILES = find_meshes_by_dirname('ycb')
    MESH_FILES.update(find_meshes_by_dirname('geom'))
    # Remove the skinny object meshes
    delete_meshes=['044_flat_screwdriver','042_adjustable_wrench']
    restrict_meshes={}
    for meshname, meshfile in MESH_FILES.items():
        mesh=get_combined_mesh(meshfile)
        if max(mesh.extents[0]/mesh.extents[1], mesh.extents[1]/mesh.extents[0])>1.7:
            delete_meshes.append(meshname)
        elif mesh.extents[2]/max(mesh.extents[0],mesh.extents[1])>1.5:
            delete_meshes.append(meshname)
            #restrict_meshes[meshname] = MESH_FILES[meshname]
    for meshname in delete_meshes:
        del MESH_FILES[meshname]
    #MESH_FILES = restrict_meshes


    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        return super()._sample_random_object_groups(dedupe_objects=True)

    def _sample_object_colors(self, num_groups):
        all_colors = [[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,0.5,0,1],[1,0,0.5,1],[0.5,0,1,1],[0,0.5,1,1],[0.5,1,0,1],[0,1,0.5,1],[0.25,0,0.5,1],[0.25,0.5,0,1],[0,0.25,0.5,1],[0.5,0.25,0,1],[0.5,0,0.25,1],[0,0.5,0.25,1]]
        # select a random color from the list of colors
        random_color_ids =  self._random_state.randint(0,len(all_colors),num_groups)
        random_colors = np.array([all_colors[i] for i in random_color_ids])
        return random_colors

    def _sample_object_meshes(self, num_groups: int) -> List[List[str]]:
       
        candidates = list(self.MESH_FILES.values())
        assert len(candidates) > 0, f"No mesh file for {self.parameters.mesh_names}."
        candidates = sorted(candidates)
        replace = True
        indices = self._random_state.choice(
            len(candidates), size=num_groups, replace=replace
        )
        return [candidates[i] for i in indices]

    def reset_goal(
        self, update_seed=False, sync_type=SyncType.RESET_GOAL
    ):
        obs = super().reset_goal(update_seed=update_seed, sync_type=sync_type, raise_when_invalid=False)
        # Check if goal placement is valid here.
        if not self._goal["goal_valid"]:
            logger.info(
                "InvalidSimulationError: "
                + self._goal.get("goal_invalid_reason", "Goal is invalid.")
            )

        return obs
    # @classmethod
    # def build_goal_generation(cls, constants, mujoco_simulation):
    #     return ObjectFixedStateGoal(
    #         mujoco_simulation,
    #         args=constants.goal_args,
    #         relative_placements=np.array(
    #             [
    #                 [0.6, 0.5],  # "029_plate"
    #                 [0.6, 0.68],  # "030_fork"
    #                 [0.6, 0.75],  # "030_fork"
    #                 [0.6, 0.36],  # "032_knife"
    #                 [0.6, 0.28],  # "031_spoon"
    #             ]
    #         ),
    #         init_quats=np.array(
    #             [
    #                 [1, 0, 0, 0],
    #                 [1, 0, 0, 0],
    #                 [1, 0, 0, 0],
    #                 [1, 0, 0, 0],
    #                 # We need to rotate the spoon a little bit counter-clock-wise to be aligned with others.
    #                 quat_from_angle_and_axis(0.38, np.array([0, 0, 1.0])),
    #             ]
    #         ),
    #     )


make_env = MyRearrangeEnv2.build
