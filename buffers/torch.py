import numpy as np
from vanguard.core import VGVecBool_T, VGVecF32_T, VGVecI64_T
from vanguard.deps import vg_assert_requirements

from veda.core.definitions import VDevice
from veda.ml.torch.algo_ppo.xp import VPPOXP, VPPOXPBatch
from veda.ml.torch.definitions import V_NUMPY_TO_TORCH_DTYPE

vg_assert_requirements("torch")
import torch


class VPPOXPTorchBuffer:
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
        "device",
        "initialized",
    )

    def __init__(self, with_capacity: int, device: VDevice):
        # NOTE(AS): Initialize variables during first append, to infer shapes
        self.states: torch.Tensor
        self.last_state: torch.Tensor
        self.actions: torch.Tensor

        self.hidden_states: torch.Tensor
        self.advantages: torch.Tensor
        self.log_prob: torch.Tensor
        self.values: torch.Tensor
        self.returns: torch.Tensor

        # NOTE(AS): Variables is not used in loss_fn, so we keep them in numpy
        self.rewards: VGVecF32_T
        self.dones: VGVecBool_T
        self.terminated: VGVecBool_T

        self.capacity = with_capacity
        self.size: int = 0
        self.initialized: bool = False
        self.device = device

    def get_batch(self, indices: VGVecI64_T) -> VPPOXPBatch:
        unravel_indices = torch.unravel_index(
            torch.from_numpy(indices).to(self.device),  # type: ignore
            self.states.shape[:2],
        )
        return VPPOXPBatch(
            states=self.states[unravel_indices],
            actions=self.actions[unravel_indices],
            advantages=self.advantages[unravel_indices],
            log_prob=self.log_prob[unravel_indices],
            returns=self.returns[unravel_indices],
            values=self.values[unravel_indices],
        )

    def _initialize_attributes(self, exp: VPPOXP):
        self.states = torch.empty(
            size=(self.capacity * exp.states.shape[0], *exp.states.shape[1:]),
            dtype=V_NUMPY_TO_TORCH_DTYPE[exp.states.dtype],
            device=self.device,
        )

        self.last_state = torch.empty(
            size=(self.capacity * exp.last_state.shape[0], *exp.last_state.shape[1:]),
            dtype=V_NUMPY_TO_TORCH_DTYPE[exp.last_state.dtype],
            device=self.device,
        )

        self.actions = torch.empty(
            size=(self.capacity * exp.actions.shape[0], *exp.actions.shape[1:]),
            dtype=V_NUMPY_TO_TORCH_DTYPE[exp.actions.dtype],
            device=self.device,
        )

        # attrubutes that always numpy, because it is needed only in recompute_adv
        self.rewards = np.empty(
            shape=(
                self.capacity * exp.rewards.shape[0],
                *exp.rewards.shape[1:],
            ),
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

        begin, end = (
            self.size * exp.states.shape[0],
            (self.size + 1) * exp.states.shape[0],
        )
        self.states[begin:end] = torch.from_numpy(exp.states).to(  # type: ignore
            self.device
        )
        self.last_state[begin:end] = torch.from_numpy(exp.last_state).to(  # type: ignore
            self.device
        )
        self.actions[begin:end] = torch.from_numpy(exp.actions).to(  # type: ignore
            self.device
        )
        # attrubutes that always numpy, because it is needed only in recompute_adv
        self.rewards[begin:end] = exp.rewards
        self.dones[begin:end] = exp.dones
        self.terminated[begin:end] = exp.terminated
        self.size += 1

    def add_epoch_targets(
        self,
        advantages: VGVecF32_T,
        log_prob: VGVecF32_T,
        values: VGVecF32_T,
        returns: VGVecF32_T,
    ):
        self.advantages = torch.from_numpy(advantages).to(  # type: ignore
            self.device
        )
        self.log_prob = torch.from_numpy(log_prob).to(  # type: ignore
            self.device
        )
        self.values = torch.from_numpy(values).to(  # type: ignore
            self.device
        )
        self.returns = torch.from_numpy(returns).to(  # type: ignore
            self.device
        )

    def num_points(self):
        return self.rewards.size

    def is_full(self):
        return self.size == self.capacity

    def clear(self):
        self.size = 0
