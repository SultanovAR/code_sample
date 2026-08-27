from typing import Literal

import numpy as np
from vanguard.core import VGVecBool_T, VGVecF32_T, VGVecI64_T
from vanguard.deps import vg_assert_requirements
from vanguard.error import vg_assert_eq

from veda.core.definitions import VDevice
from veda.ml.torch.algo_ppo.xp import VPPOXP, VPPOXPBatch

vg_assert_requirements("torch")
import torch


class VPPOXPBuffer:
    __slots__ = (
        "states",
        "last_state",
        "actions",
        "rewards",
        "dones",
        "terminated",
        "advantages",
        "log_prob",
        "values",
        "returns",
        #
        "capacity",
        "size",
        "initialized",
    )

    def __init__(self, with_capacity: int, device: Literal[VDevice.CPU]):
        # NOTE(AS): Initialize variables during first append, to infer shapes
        vg_assert_eq(
            device,
            VDevice.CPU,
            "VPPOXPBuffer can't be used with a non-cpu device! Use VPPOXPTorchBuffer instead!",
        )

        self.states: VGVecF32_T
        self.last_state: VGVecF32_T
        self.actions: VGVecF32_T
        self.rewards: VGVecF32_T
        self.dones: VGVecBool_T
        self.terminated: VGVecBool_T

        self.advantages: VGVecF32_T
        self.log_prob: VGVecF32_T
        self.values: VGVecF32_T
        self.returns: VGVecF32_T

        self.capacity = with_capacity
        self.size: int = 0
        self.initialized: bool = False

    def get_batch(self, indices: VGVecI64_T) -> VPPOXPBatch:
        unravel_indices = np.unravel_index(indices, shape=(self.states.shape[:2]))
        return VPPOXPBatch(
            torch.from_numpy(self.states[unravel_indices]),  # type: ignore
            torch.from_numpy(self.actions[unravel_indices]),  # type: ignore
            torch.from_numpy(self.advantages[unravel_indices]),  # type: ignore
            torch.from_numpy(self.log_prob[unravel_indices]),  # type: ignore
            torch.from_numpy(self.returns[unravel_indices]),  # type: ignore
            torch.from_numpy(self.values[unravel_indices]),  # type: ignore
        )

    def _initialize_attributes(self, exp: VPPOXP):
        self.states = np.empty(
            shape=(self.capacity * exp.states.shape[0], *exp.states.shape[1:]),
            dtype=exp.states.dtype,
        )

        self.last_state = np.empty(
            shape=(self.capacity * exp.last_state.shape[0], *exp.last_state.shape[1:]),
            dtype=exp.last_state.dtype,
        )

        self.actions = np.empty(
            shape=(self.capacity * exp.actions.shape[0], *exp.actions.shape[1:]),
            dtype=exp.actions.dtype,
        )

        self.rewards = np.empty(
            shape=(self.capacity * exp.rewards.shape[0], *exp.rewards.shape[1:]),
            dtype=exp.rewards.dtype,
        )

        self.dones = np.empty(
            shape=(self.capacity * exp.dones.shape[0], *exp.dones.shape[1:]),
            dtype=exp.dones.dtype,
        )

        self.terminated = np.empty(
            shape=(
                self.capacity * exp.terminated.shape[0],
                *exp.terminated.shape[1:],
            ),
            dtype=exp.terminated.dtype,
        )

    def extend(self, exp: VPPOXP):
        if not self.initialized:
            self._initialize_attributes(exp)
            self.initialized = True

        # NOTE(AS): fill created arrays on certain indicies
        begin, end = (
            self.size * exp.states.shape[0],
            (self.size + 1) * exp.states.shape[0],
        )
        # fmt: off
        self.states    [begin : end] = exp.states
        self.last_state[begin : end] = exp.last_state
        self.actions   [begin : end] = exp.actions

        self.rewards    [begin : end] = exp.rewards
        self.dones      [begin : end] = exp.dones
        self.terminated [begin : end] = exp.terminated
        # fmt: on
        self.size += 1

    def add_epoch_targets(
        self,
        advantages: VGVecF32_T,
        log_prob: VGVecF32_T,
        values: VGVecF32_T,
        returns: VGVecF32_T,
    ):
        self.advantages = advantages
        self.log_prob = log_prob
        self.values = values
        self.returns = returns

    def num_points(self):
        return self.rewards.size

    def is_full(self):
        return self.size == self.capacity

    def clear(self):
        self.size = 0
