# pyright: reportPrivateImportUsage=false
import numpy as np

from typing import Any, Iterator, List, Mapping

from vanguard import logger
from vanguard.core import (
    VGPath,
    VGVecF32_T,
    VGVecBool_T,
    u32,
    vgretry,
)
from vanguard.deps import vg_assert_requirements
from vanguard.error import VGInvariantError
from vanguard.stash import VGFileStash
from voyager2.onnx.definitions import VG_ONNXMOD_EXPORT_NAME
from voyager2.onnx.exporter import bender_figment_name
from voyager2.runtime import VGRuntimeCtx
from voyager2.typelib.ctors import Limit
from voyager2.typelib.definitions import MustHave
from vanguard.types import vg_type_cast
from vanguard.meta import vg_rebind_method
from voyager2.factory import vg_create
from voyager2.descriptions.toml import TOMLDescription

from veda.core.config import CommonNetConfig
from veda.core.ctx import VTrainingCtx
from veda.core.definitions import (
    IModuleMetadata,
    VBasicXP,
    VDevice,
    VAdapters,
    IAgent,
)
from veda.core.sched import VedaSchedulerConfig, VedaScheduler
from veda.core.rms import VRunningMeanStd
from veda.pipeline_config import PipelineConfig

vg_assert_requirements("torch")
import torch

from veda.ml.torch.onnx import VTorchExportArguments, VTorchExporter
from veda.ml.torch.definitions import VTorchNetwork
from veda.ml.torch.config import LRSchedulerConfig, VOptimizerConfig
from veda.ml.torch.net.config import VNetworkConfig_T
from veda.ml.torch.net.utils import update_parameter_groups
from veda.ml.torch.net.conv import VCNNConfig
from veda.ml.torch.net.tcn import TCNNConfig
from veda.ml.torch.net.mlp import VMLPConfig
from veda.ml.torch.algo_ppo.net_init import vppo_weight_init
from veda.ml.torch.algo_ppo.buffers.numpy import VPPOXPBuffer
from veda.ml.torch.algo_ppo.buffers.torch import VPPOXPTorchBuffer
from veda.ml.torch.algo_ppo.xp import (
    VPPOXP,
    VPPOXPBatch,
    v_new_ppo_buffer,
)
from veda.ml.torch.algo_ppo.actor import VPPOActor, PPOActorConfig
from veda.ml.torch.utils import (
    v_apply_norm_,
    v_gae_return,
    v_report_grad_metrics,
    v_symlog,
    v_bt_flat,
)


def _fill_cfg_input_shapes(cfg: "PPOAgentConfig", input_shape: tuple[int, ...]) -> None:
    """Fills empty input_shape fields in the network configuration.

    This function is executed at runtime because game attributes
    are only accessible during runtime.
    """

    def fill_if_empty(cfg: CommonNetConfig, attr: str, value: int):
        if getattr(cfg, attr) is None:
            setattr(cfg, attr, value)

    match cfg.backbone:
        case VMLPConfig():
            fill_if_empty(cfg.backbone, "input_size", input_shape[0])
        case VCNNConfig():
            fill_if_empty(cfg.backbone, "input_channels", input_shape[0])
            fill_if_empty(cfg.backbone, "input_height", input_shape[1])
            fill_if_empty(cfg.backbone, "input_width", input_shape[2])
        case TCNNConfig():
            fill_if_empty(cfg.backbone, "input_size", input_shape[-1])
        case _:
            fill_if_empty(cfg.actor.network, "input_size", input_shape[0])
            fill_if_empty(cfg.critic, "input_size", input_shape[0])


@VGRuntimeCtx.vgconfig_name("torch.agent", "ppo")
class PPOAgentConfig(TOMLDescription):
    # fmt: off
    max_grad_norm = Limit[float].default(0.0, min=0.0, max=1.0)
    veda_scheduler: VedaSchedulerConfig = VedaSchedulerConfig()
    optimizer: MustHave[VOptimizerConfig]

    actor:     MustHave[PPOActorConfig ]
    critic:    MustHave[VNetworkConfig_T]

    backbone: VNetworkConfig_T | None = None
    lr_scheduler: LRSchedulerConfig | None = None
    check_grads: bool = True

    seq_len = Limit[int].default(0, min=0)
    buf_dim: MustHave[int]
    """Controls the number of dimensions in the PPOBuffer"""

    # NOTE(AS): PPO args
    gamma        = Limit[float].default(0.99, min=1e-8, max=1.0)
    gae_lambda   = Limit[float].default(0.95, min=1e-8, max=1.0)
    policy_clip  = Limit[float].default(0.2, min=0, max=1)

    critic_loss_coefficient = Limit[float].default(0.1, min=0, max=1)
    entropy_coefficient     = Limit[float].default(0, min=0, max=1)
    number_of_updates       = Limit[int].default(2, min=1, max=10)
    num_reward_clusters     = Limit[int].default(-1)

    value_clip: bool     = False
    norm_adv: bool       = True
    recompute_adv: bool  = False
    state_norm: bool     = False
    return_norm: bool    = False
    symlog_rewards: bool = False
    symlog_states:  bool = False
    dpo_loss: bool = False

    sampling_on_validation: bool = False
    prop_init: bool = False
    temperature: float = Limit[float].default(1.0, min=0.0)
    # fmt: on


@VGRuntimeCtx.vgbind_config(PPOAgentConfig)
class VPPOAgent(
    IAgent[torch.Tensor, VPPOXP, VPPOXPBatch, VPPOActor],
):
    __slots__ = (
        "__ctx",
        "adapters",
        "actor",
        "critic",
        "optimizer",
        "lr_scheduler",
        "v_sched",
        # hyperparams
        "gamma",
        "gae_lambda",
        "policy_clip",
        "value_clip",
        "norm_adv",
        "check_grads",
        "recompute_adv",
        "max_grad_norm",
        "symlog_rewards",
        "symlog_states",
        "dpo_loss",
        "critic_loss_coefficient",
        "entropy_coefficient",
        "number_of_updates",
        "num_reward_clusters",
        "sampling_on_validation",
        "temperature",
        "seq_len",
        "state_rms",
        "return_rms",
        "_optim_params",
        "export_args",
        "exp_storage",
        "__stash",
        "__onnx_prefix",
    )

    def __init__(
        self,
        config: PPOAgentConfig,
        adapters: VAdapters[torch.Tensor],
        ctx: VTrainingCtx[PipelineConfig],
    ):
        self.__ctx = ctx
        if self.__ctx.device == VDevice.MPS:
            logger.warning("Compilation for MPS is unavaliable, using CPU instead")
            self.__ctx.device = VDevice.CPU
        self.adapters = adapters

        # NETWORK
        _fill_cfg_input_shapes(config, ctx.input_shape)
        if config.backbone is None:
            actor_backbone_net = None
        else:
            actor_backbone_net = vg_create[VTorchNetwork].new(config.backbone).unwrap()
        self.actor = VPPOActor(
            **config.actor,
            backbone_network=actor_backbone_net,
            is_discrete=adapters.action.is_discrete(),
            n_actions=adapters.action.get_n_actions(),
            ndim_act=adapters.action.get_ndim(),
            symlog_states=config.symlog_states,
            temperature=config.temperature,
        )
        self.actor.apply(vppo_weight_init)
        if config.prop_init:
            self.actor.action_head.model[-1].weight.data.copy_(  # type: ignore
                0.01 * self.actor.action_head.model[-1].weight.data  # type: ignore
            )
        self.actor.set_device(ctx.device)

        self.critic = vg_create[VTorchNetwork].new(config.critic).unwrap()
        self.critic.apply(vppo_weight_init)
        self.critic.to(ctx.device)
        # OPTIMIZER AND SCHEDULERS
        param_groups: list[Mapping[str, Any]] = []

        # NOTE(AS): backbone params already in self.actor
        update_parameter_groups("policy", self.actor, param_groups)
        update_parameter_groups("critic", self.critic, param_groups)
        opt_dict = config.optimizer.to_dict()
        opt_dict["lr"] = torch.as_tensor(opt_dict["lr"], device=ctx.device)  # type: ignore
        self.optimizer = (
            vg_create[torch.optim.Optimizer]
            .new_from(
                params=param_groups,
                **opt_dict,  # type: ignore
            )
            .unwrap()
        )
        self.lr_scheduler = (
            vg_create[torch.optim.lr_scheduler.LRScheduler]
            .new_from(**config.lr_scheduler, optimizer=self.optimizer)
            .unwrap()
            if config.lr_scheduler
            else None
        )
        self.v_sched = VedaScheduler(config=config.veda_scheduler)

        # HYPERPARAMETRS
        # fmt: off
        self.gamma       = config.gamma
        self.gae_lambda  = config.gae_lambda
        # Note(vb): to avoid recompilation when using veda scheduler, see issue https://github.com/pytorch/pytorch/issues/120934
        self.policy_clip = torch.as_tensor(config.policy_clip, device=self.__ctx.device)
        self.value_clip  = torch.as_tensor(config.value_clip, device=self.__ctx.device)
        self.temperature = torch.as_tensor(config.temperature, device=self.__ctx.device)
        self.norm_adv    = config.norm_adv
        self.check_grads = config.check_grads

        self.recompute_adv  = config.recompute_adv
        self.max_grad_norm  = config.max_grad_norm
        self.symlog_rewards = config.symlog_rewards
        self.symlog_states  = config.symlog_states
        self.dpo_loss = config.dpo_loss

        self.critic_loss_coefficient = torch.as_tensor(config.critic_loss_coefficient, device=self.__ctx.device)
        self.entropy_coefficient     = torch.as_tensor(config.entropy_coefficient, device=self.__ctx.device) #Note(VB): for veda scheduler
        self.number_of_updates       = config.number_of_updates
        self.num_reward_clusters     = config.num_reward_clusters
        self.sampling_on_validation  = config.sampling_on_validation

        self.seq_len = max(config.seq_len, 1)

        # OTHERS
        self.state_rms   = VRunningMeanStd() if config.state_norm else None
        self.return_rms  = VRunningMeanStd() if config.return_norm else None
        self.exp_storage = v_new_ppo_buffer(config.buf_dim, self.__ctx.device)

        # fmt: on

        self.export_args = VTorchExportArguments(
            args=(
                torch.randn(
                    size=self.actor.get_input_shape(),
                    dtype=torch.float32,
                    device=self.__ctx.device,
                ).repeat_interleave(self.seq_len, 0),
                # NOTE(AS): export -> (1, n), metadata -> (n,)
                torch.zeros(
                    size=(1, *self.actor.get_state().shape),
                    dtype=torch.float32,
                    device=self.__ctx.device,
                ),
            ),
            opset_version=20,
        )

        # ONNX
        self.__stash = VGFileStash("torch-agent")
        self.__onnx_prefix = (
            VGRuntimeCtx.config.storage.local_dir
            / f"{self.__class__.__name__}_{self.actor._get_name()}"  # pyright: ignore[reportPrivateUsage]
        )

        # OPTIMIZATION(iy): rebinds for backbone
        if self.actor.backbone_network is not None:
            vg_rebind_method(self._prepare_states, self._prepare_states_backbone)
            vg_rebind_method(self.send_grad_metrics, self.send_grad_metrics_backbone)

        # Note(VB): _optim params is an attribute for max_grad_norm. Save pointers for params before training.
        self._optim_params: List[torch.nn.parameter.Parameter] = [
            param
            for group in self.optimizer.param_groups
            for param in group["params"]
            if param.requires_grad
        ]

    #################
    # HOOKS
    #################

    def on_train_start(self) -> None:
        if self.state_rms and self.exp_storage.is_full():
            match self.exp_storage:
                case VPPOXPBuffer():
                    self.state_rms.update(self.exp_storage.states[0])
                case VPPOXPTorchBuffer():
                    self.state_rms.update(self.exp_storage.states[0].cpu().numpy())  # type: ignore
            self.actor.set_states_statistics(
                mean=self.state_rms.mean,
                var=self.state_rms.var,
                clip=self.state_rms.clip_max,
            )
            self.state_rms.reset()
        self.exp_storage.clear()
        self.__ctx.global_step = 0

    def on_train_end(self) -> None:
        pass

    def on_epoch_start(self):
        self.exp_storage.clear()
        self.actor.sample_action = True
        self.actor.eval()

    def before_train_steps(self) -> None:
        if self.state_rms:
            match self.exp_storage:
                case VPPOXPBuffer():
                    self.state_rms.update(v_bt_flat(self.exp_storage.states))
                    self.state_rms.norm_(self.exp_storage.states)
                    self.state_rms.norm_(self.exp_storage.last_state)
                case VPPOXPTorchBuffer():
                    self.state_rms.update(
                        v_bt_flat(self.exp_storage.states.cpu().numpy())  # type: ignore
                    )
                    mean = torch.from_numpy(self.state_rms.mean).to(self.__ctx.device)  # type: ignore
                    std = torch.from_numpy(self.state_rms.std()).to(self.__ctx.device)  # type: ignore
                    clip_max = (
                        torch.tensor(self.state_rms.clip_max)
                        .float()
                        .to(self.__ctx.device)
                    )
                    v_apply_norm_(
                        self.exp_storage.states,
                        mean=mean,
                        std=std,
                        clip=clip_max,
                    )
                    v_apply_norm_(
                        self.exp_storage.last_state,
                        mean=mean,
                        std=std,
                        clip=clip_max,
                    )

        self.add_ppo_targets_to_buffer()
        self.critic.train()
        self.actor.train()

    def on_epoch_end(self) -> None:
        # NOTE(AS):
        # send statistics to the actor so that network
        # gets consistent vectors on experience collection (ONNX)
        # and on training
        if self.state_rms:
            self.actor.set_states_statistics(
                mean=self.state_rms.mean,
                var=self.state_rms.var,
                clip=self.state_rms.clip_max,
            )

        for param in self.v_sched.cfg.keys():
            current_value = getattr(self, param)
            VGRuntimeCtx.io.report_metric(
                key=b"~_last_value",
                tag=param.encode(),
                value=current_value,
                step=self.__ctx.global_step,
            )
            setattr(
                self,
                param,
                self.v_sched.step(param, current_value, self.__ctx.epoch_counter + 1),
            )

        if "temperature" in self.v_sched.cfg.keys():
            self.actor.action_head.set_temperature(self.temperature)

        if self.lr_scheduler:
            VGRuntimeCtx.io.report_metric(
                key=b"~_last_value",
                tag=b"learning_rate",
                value=self.lr_scheduler.get_last_lr()[0],
                step=self.__ctx.global_step,
            )
            self.lr_scheduler.step()

        logger.debug("Removing temporary simulation figments")
        self.__stash.forget(remove_files=True)
        self.__ctx.epoch_counter += 1

    def on_valid_start(self) -> None:
        self.actor.sample_action = self.sampling_on_validation
        self.actor.eval()

    def on_valid_end(self) -> None:
        pass

    #################
    # EXPERIENCE
    #################

    def _prepare_states_backbone(
        self, states: VGVecF32_T | torch.Tensor
    ) -> torch.Tensor:
        # NOTE(iy): Invariant is checked in __init__
        with torch.no_grad():
            return self.actor.backbone_network(
                torch.as_tensor(states, device=self.__ctx.device)
            )  # type: ignore

    def _prepare_states(self, states: VGVecF32_T | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(states, device=self.__ctx.device)

    def _compute_returns(
        self,
        states: torch.Tensor,
        last_state: torch.Tensor,
        rewards: VGVecF32_T,
        dones: VGVecBool_T,
        terminateds: VGVecBool_T,
    ) -> tuple[VGVecF32_T, VGVecF32_T, VGVecF32_T]:
        """
        Args:
            states: (batch*time, features)
            last_state: (batch, features)
            rewards: (batch, time)
            dones: (batch, time)
            terminateds: (batch, time)
        Returns:
            returns: (batch, time)
            advantages: (batch, time)
            values: (batch, time)
        """
        with torch.no_grad():
            values = self.critic(states).cpu().numpy()  # type: ignore
            values = values.reshape(rewards.shape)  # (batch, time)
            last_value = self.critic(last_state).cpu().numpy()  # type: ignore
            last_value = last_value.squeeze(1)
            next_values = np.concatenate(
                [values[:, 1:], last_value[:, np.newaxis]], axis=1
            )
            next_values = next_values * (1 - terminateds.astype(np.float32))

        if self.return_rms:
            values = self.return_rms.descale(values)
            next_values = self.return_rms.descale(next_values)

        advantages = v_gae_return(
            values,
            next_values,
            rewards,
            dones,
            np.float32(self.gamma),
            np.float32(self.gae_lambda),
        )

        unnormalized_returns = advantages + values
        if self.return_rms:
            returns = self.return_rms.scale(unnormalized_returns)
            self.return_rms.update(v_bt_flat(unnormalized_returns))
        else:
            returns = unnormalized_returns
        return returns, advantages, values

    def process_xp(self, xp: VBasicXP) -> VPPOXP:
        if self.seq_len > 1:
            # NOTE(tf): exp. for trading tasks, comes with shape (1, ...), that's why we keep that single dimension
            xp.states = xp.states.reshape(1, xp.states.shape[1], self.seq_len, -1)[
                :, self.seq_len :
            ]
            xp.actions = xp.actions[:, self.seq_len :]
            xp.rewards = xp.rewards[:, self.seq_len :]
            xp.dones = xp.dones[:, self.seq_len :]
            xp.terminated = xp.terminated[:, self.seq_len :]

        xp = VPPOXP(xp.states, xp.actions, xp.rewards, xp.dones, xp.terminated)
        if self.symlog_states:
            xp.states = v_symlog(xp.states)
        if self.symlog_rewards:
            xp.rewards = v_symlog(xp.rewards)
        xp.actions = xp.actions.reshape(
            *xp.rewards.shape, self.actor.action_head.ndim_act
        )
        self.exp_storage.extend(xp)
        return xp

    def add_ppo_targets_to_buffer(self):
        if not self.exp_storage.is_full():
            raise VGInvariantError(
                f"size={self.exp_storage.size}, capacity={self.exp_storage.capacity},"
                "but you should call self.get_batch(...) only if size == capacity,"
                "otherwise you will use unintialized data"
            )

        f_states = self._prepare_states(v_bt_flat(self.exp_storage.states))
        f_last_state = self._prepare_states(self.exp_storage.last_state)
        returns, advantages, values = self._compute_returns(
            states=f_states,
            last_state=f_last_state,
            rewards=self.exp_storage.rewards,
            dones=self.exp_storage.dones,
            terminateds=self.exp_storage.terminated,
        )

        b_dim, t_dim, a_dim = self.exp_storage.actions.shape
        with torch.no_grad():
            log_prob: VGVecF32_T = vg_type_cast(
                self.actor.get_dist(f_states)
                .log_prob(
                    torch.as_tensor(
                        self.exp_storage.actions.reshape(b_dim * t_dim, a_dim),
                        device=self.__ctx.device,
                    )
                )
                .cpu()
                .numpy()  # type: ignore
                .reshape(b_dim, t_dim)
            )
        self.exp_storage.add_epoch_targets(
            advantages=advantages, log_prob=log_prob, values=values, returns=returns
        )

    def iter_batches(self) -> Iterator[VPPOXPBatch]:
        for num_update in range(self.number_of_updates):
            if num_update > 0 and self.recompute_adv:
                returns, advantages, values = self._compute_returns(
                    self._prepare_states(v_bt_flat(self.exp_storage.states)),
                    self._prepare_states(self.exp_storage.last_state),
                    self.exp_storage.rewards,
                    self.exp_storage.dones,
                    self.exp_storage.terminated,
                )
                if self.__ctx.device != VDevice.CPU:
                    returns = torch.as_tensor(returns, device=self.__ctx.device)
                    advantages = torch.as_tensor(advantages, device=self.__ctx.device)
                    values = torch.as_tensor(values, device=self.__ctx.device)

                self.exp_storage.returns = returns  # type: ignore
                self.exp_storage.advantages = advantages  # type: ignore
                self.exp_storage.values = values  # type: ignore

            indices = np.random.permutation(
                self.__ctx.batches_per_epoch * self.__ctx.batch_size
            ).reshape(self.__ctx.batches_per_epoch, self.__ctx.batch_size)
            for idx_vec in indices:
                yield self.exp_storage.get_batch(idx_vec)

    ##################
    # TRAIN
    ##################

    @torch.compile(mode="reduce-overhead")  # type: ignore
    def update_network(self, batch: VPPOXPBatch) -> dict[str, torch.Tensor]:
        if self.actor.backbone_network is not None:
            batch.states = self.actor.backbone_network(batch.states)
        loss_dict = self.loss_fn(batch)
        loss = loss_dict["total_loss"]
        loss.backward()  # type: ignore
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self._optim_params, self.max_grad_norm)
        self.optimizer.step()
        return loss_dict

    def train_step(self, batch: VPPOXPBatch) -> None:
        # NOTE(vb): free cudagraph from previous step computations
        torch.compiler.cudagraph_mark_step_begin()
        # NOTE(iy): set_to_none should compute faster
        self.optimizer.zero_grad(set_to_none=True)
        loss_dict = self.update_network(batch)
        for loss_key, loss_value in loss_dict.items():
            VGRuntimeCtx.io.stash_metric(
                b"01_loss",
                loss_value.item(),
                tag=loss_key.encode(),
                step=self.__ctx.global_step,
            )
        if self.check_grads:
            self.send_grad_metrics()
        self.__ctx.global_step += 1

    @torch.compile(mode="reduce-overhead")  # type: ignore
    def loss_fn(self, batch: VPPOXPBatch) -> dict[str, torch.Tensor]:
        # NOTE(AS): normalizing advantages
        advs = batch.advantages
        if self.norm_adv:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # NOTE(AS): actor loss
        dist = self.actor.get_dist(batch.states)
        new_logprobs = dist.log_prob(batch.actions)

        if self.dpo_loss:
            log_diff = new_logprobs - batch.log_prob
            ratios = log_diff.exp()
            positive_adv = torch.gt(advs, 0.0)
            r1 = ratios - 1.0
            drift1 = torch.nn.functional.relu(
                r1 * advs - 2.0 * torch.tanh(r1 * advs / 2.0)
            )
            drift2 = torch.nn.functional.relu(
                log_diff * advs - 0.6 * torch.tanh(log_diff * advs / 0.6)
            )
            drift = drift1 * positive_adv + drift2 * (1 - positive_adv)
            clip_loss = -(ratios * advs - drift).mean()
        else:
            ratios = (new_logprobs - batch.log_prob).exp()
            surr1 = ratios * advs
            surr2 = ratios.clamp(1.0 - self.policy_clip, 1.0 + self.policy_clip) * advs
            clip_loss = -torch.min(surr1, surr2).mean()

        # NOTE(AS): value loss
        value = self.critic(batch.states).flatten()
        if self.value_clip:
            v_clip = batch.values + (value - batch.values).clamp(
                -self.policy_clip,
                self.policy_clip,
            )
            vf1: torch.Tensor = (batch.returns - value).pow(2)
            vf2: torch.Tensor = (batch.returns - v_clip).pow(2)
            vf_loss = torch.mean(torch.max(vf1, vf2))
        else:
            vf_loss = torch.mean((batch.returns - value).pow(2))

        # NOTE(AS): final loss
        entropy: torch.Tensor = torch.mean(dist.entropy())
        entropy_loss = self.entropy_coefficient * entropy  # to maximize
        critic_loss = self.critic_loss_coefficient * vf_loss
        loss: torch.Tensor = clip_loss + critic_loss - entropy_loss

        # NOTE(AS): logging
        approx_kl = ((ratios - 1) - torch.log(ratios)).mean().detach()
        clip_frac = ((ratios - 1.0).abs() > self.policy_clip).float().mean().detach()

        loss_dict = {
            "total_loss": loss,
            "actor_loss": clip_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss,
            "kl_div": approx_kl,
            "clip_frac": clip_frac,
        }
        return loss_dict

    ##############
    # LOGGING
    ##############

    def send_grad_metrics(self) -> None:
        v_report_grad_metrics(self.actor.network, "Actor", self.__ctx.global_step)
        v_report_grad_metrics(self.critic, "Critic", self.__ctx.global_step)

    def send_grad_metrics_backbone(self) -> None:
        v_report_grad_metrics(self.actor.network, "Actor", self.__ctx.global_step)
        v_report_grad_metrics(self.critic, "Critic", self.__ctx.global_step)
        v_report_grad_metrics(self.actor.backbone_network, "BB", self.__ctx.global_step)  # type: ignore

    ##############
    # ONNX
    ##############

    def new_onnx_exporter(self) -> VTorchExporter:
        return VTorchExporter(
            model=self.actor,
            args=self.export_args,
            precompile=False,
            dynamic_axes=False,
        )

    # FIXME(iy): API for to_onnx looks bad for agent imho.
    @vgretry(10)
    def to_onnx(self) -> VGPath:
        return VTorchExporter(
            model=self.actor,
            args=self.export_args,
            precompile=False,
            dynamic_axes=False,
        ).export(
            VGPath(
                f"{self.__onnx_prefix}@{self.__ctx.global_step}.{VG_ONNXMOD_EXPORT_NAME}"
            )
        )

    @vgretry(10)
    def to_onnx_seeded(
        self, n_models: u32, metadata: type[IModuleMetadata]
    ) -> list[VGPath]:
        """Converts inner network into ONNX graph.
        Args:
            n_models: Number of models to export (with random seeds baked into them)

        Returns
            local onnx file paths
        """
        path = VGPath(f"{self.__onnx_prefix}@{self.__ctx.global_step}_train")

        logger.debug("Exporting models with seed:")
        exporter = VTorchExporter(
            model=self.actor,
            args=self.export_args,
            precompile=False,
            dynamic_axes=False,
        )
        ref_onnx = exporter.export(bender_figment_name(path))
        self.__stash.remember(ref_onnx)

        result = [exporter.bake_seed_info(ref_onnx) for _ in range(n_models)]
        self.__stash.extend(result)

        # NOTE(iy): Support
        if not metadata.is_required:
            return result

        result = [
            exporter.patch_metadata(seeded, metadata.from_actor(self.actor))
            for seeded in result
        ]
        self.__stash.extend(result)
        return result
