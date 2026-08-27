from vanguard.core import VGVecF32_T
from vanguard.deps import vg_assert_requirements
from vanguard.error import VGAttributeError
from vanguard.meta import vg_rebind_method
from vanguard.types import vg_type_cast
from voyager2.descriptions.toml import TOMLDescription
from voyager2.factory import vg_create
from voyager2.typelib.definitions import MustHave

vg_assert_requirements("torch")
import numpy as np
import torch

from veda.core.definitions import VDevice
from veda.ml.torch.definitions import (
    VTorchActor,
    VTorchNetwork,
    VTorchNetworkOutput,
)
from veda.ml.torch.net.action_head import VContinuousActionHead, VDiscreteActionHead
from veda.ml.torch.net.config import VNetworkConfig_T


class PPOActorContinuousArgsConfig(TOMLDescription):
    conditioned_sigma: bool = False
    bound_mu: bool = False


class PPOActorConfig(TOMLDescription):
    network: MustHave[VNetworkConfig_T]
    continuous_arg: PPOActorContinuousArgsConfig = PPOActorContinuousArgsConfig()


V_EMPTY_TENSOR = torch.zeros((1, 1), dtype=torch.float32)
V_EMPTY_VECTOR = np.zeros((1, 1), dtype=np.float32)


class VPPOActor(VTorchActor):
    def __init__(
        self,
        is_discrete: bool,
        n_actions: int,
        ndim_act: int,
        network: VNetworkConfig_T,
        # TODO(AS): TODO signature of backbone with hidden_states
        backbone_network: VTorchNetwork | None,
        continuous_arg: PPOActorContinuousArgsConfig,
        symlog_states: bool,
        temperature: float,
    ):
        super().__init__()  # type: ignore
        if not network.output_size:
            raise VGAttributeError(
                "actor.network.config doens't have output_size attribute",
                "check autoinference or pipeline config",
            )
        self.backbone_network = backbone_network
        self.network = vg_create[VTorchNetwork].new(network).unwrap()

        if is_discrete:
            self.action_head = VDiscreteActionHead(
                input_size=network.output_size,
                n_actions=n_actions,
                ndim_act=ndim_act,
                use_bias=True,
                reparam_net=False,
                layer_norm=False,
                temperature=temperature,
            )
        else:
            self.action_head = VContinuousActionHead(
                input_size=network.output_size,
                n_actions=n_actions,
                ndim_act=ndim_act,
                conditioned_sigma=continuous_arg.conditioned_sigma,
                bound_mu=continuous_arg.bound_mu,
                use_bias=True,
                reparam_net=False,
                layer_norm=False,
                temperature=temperature,
            )

        self.register_buffer("dummy_info", torch.empty((1, 0), dtype=torch.float32))
        self.register_buffer("states_mean", torch.tensor([0.0]).float())
        self.register_buffer("states_var", torch.tensor([1.0]).float())
        self.register_buffer("clip_max", torch.tensor(0.0).float())
        self.sample_action = True
        self.symlog_states = symlog_states

        if self.backbone_network:
            vg_rebind_method(self.get_input_shape, self._get_input_shape_backbone)

    def get_state(self) -> VGVecF32_T:
        return V_EMPTY_VECTOR.flatten()

    def _get_input_shape_backbone(self) -> tuple[int, ...]:
        return self.backbone_network.input_shape  # type: ignore

    def get_input_shape(self) -> tuple[int, ...]:
        return self.network.input_shape  # type: ignore

    def get_addinfo_shape(self) -> tuple[int, ...]:
        return vg_type_cast(self.dummy_info.shape)

    def set_states_statistics(
        self, mean: VGVecF32_T, var: VGVecF32_T, clip: np.float32
    ):
        self.states_mean = torch.from_numpy(mean).float().to(self.device)  # type: ignore
        self.states_var = torch.from_numpy(var).float().to(self.device)  # type: ignore
        self.clip_max = torch.tensor(clip, dtype=torch.float32, device=self.device)  # type: ignore

    def set_device(self, device: VDevice):
        self.device = device
        self.to(self.device)

    # @torch.no_grad()  # type: ignore
    def forward(
        self, input: torch.Tensor, states: torch.Tensor = V_EMPTY_TENSOR
    ) -> VTorchNetworkOutput:
        """
        Returns actions based on self.backbone_net, self.network and states
        Args:
            states: input to the network, tensor with dims [batch, feature_size]
        Returns:
            (additional_info, action): (Tensor, Tensor)
        """
        # Note(vb):
        # Torch version of symlog states regime from Dreamer V3.
        if self.symlog_states:
            input = torch.sign(input) * (input.abs() + 1).log()
        # Note (AS):
        # reimplementing veda.ml.statistics.RunningMeanStd.norm()
        # for similar inputs into net on experience collection and training
        input = (input - self.states_mean) / torch.sqrt(self.states_var + 1e-8)
        if self.clip_max:
            input = torch.clip(input, -self.clip_max, self.clip_max)

        if self.backbone_network is not None:
            input = self.backbone_network(input)
        dist = self.get_dist(input)
        action: torch.Tensor = vg_type_cast(
            dist.sample() if self.sample_action else dist.mode
        )
        # NOTE(AS):
        # Why squeeze() is needed:
        #   trading.experience:
        #       input: (1, feat_size), after sample: (1, ndim_act), need: (any_size,)
        #   game.experience:
        #       input: (batch, feat_size), after sample: (batch, ndim_act), need: (batch, ndim_act)
        # NOTE(AS):
        # Don't forget to reshape in back (batch, ndim_act) in loss before dist.log_prob()
        return VTorchNetworkOutput(
            torch.tile(self.dummy_info, (input.shape[0], 1)),  # type: ignore
            action.squeeze(0).float(),
            states,
        )

    def get_dist(self, states: torch.Tensor) -> torch.distributions.Distribution:
        """
        Returns distribution

        NOTE:
            backbone_network forward won't be called in this method, make sure you called it if needed

        Args:
            states: input to the network
        Returns:
            torch.distributions.Categorical
        """
        logits = self.network.forward(states)
        return self.action_head.get_dist(logits)
