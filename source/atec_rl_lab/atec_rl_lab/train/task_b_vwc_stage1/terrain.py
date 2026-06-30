from copy import deepcopy

from atec_rl_lab.tasks.task_b.terrain import TASK_B_TERRAIN_CFG


TASK_B_STAGE1_TERRAIN_CFG = deepcopy(TASK_B_TERRAIN_CFG)
TASK_B_STAGE1_TERRAIN_CFG.terrain_generator.num_rows = 9
TASK_B_STAGE1_TERRAIN_CFG.terrain_generator.num_cols = 9
