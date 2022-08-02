from __future__ import annotations

from copy import deepcopy
import opentuner as ot # type: ignore
import numpy as np
from opentuner.search.manipulator import BooleanParameter, BooleanArray # type: ignore
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import MSCCLProgram
from .ir import chan_t, get_num_chunks
from .simulator import build_world


class MyBooleanArray(BooleanArray):
    def add_difference(self, cfg_dst: np.ndarray, scale, cfg_b: np.ndarray, cfg_c: np.ndarray):
        new_val = np.logical_or(self._get(cfg_dst), np.logical_xor(self._get(cfg_b), self._get(cfg_c)))
        self._set(cfg_dst, new_val)


class ScheduleTuner(ot.MeasurementInterface):

    def __init__(self, program: MSCCLProgram, size: int, dest: dict, *args):
        self.program = program
        self.size = size
        self.dest = dest
        super(ScheduleTuner, self).__init__(*args)

    def manipulator(self):
        man = ot.ConfigurationManipulator()

        man.add_parameter(ot.IntegerParameter('num_instances', 1, 16))
        man.add_parameter(ot.IntegerParameter('total_connections', 1, 32)) # this is num_instances * num_channels
        man.add_parameter(BooleanParameter('channels_blocked')) # whether or not to use blocked-mode channel assignment
        man.add_parameter(MyBooleanArray('merged_threadblocks', 32)) # bitvector representing which channels' threadblocks to merge

        return man

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data

        # print(cfg)

        num_instances = cfg['num_instances']
        num_channels = max(1, cfg['total_connections'] // cfg['num_instances'])
        blocked = cfg['channels_blocked']
        tb_merges = {chan_t(c) for c in range(num_channels) if cfg['merged_threadblocks'][0][c]}

        if (prog := deepcopy(self.program).parameterized_schedule(num_instances, num_channels, blocked, tb_merges)) is None:
            return ot.Result(time=float('inf'))
        
        chunks = get_num_chunks(prog)

        world = build_world(prog, chunksize=self.size / chunks)
        world.initialize()
        
        return ot.Result(time=world.run())

    def save_final_config(self, config):
        self.dest.update(config.data)


