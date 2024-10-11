from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
import torch
from torch.nn import functional as F

from a2c import MODEL_REWARD_PREDS, MODEL_VF_PREDS
from a2c.a2c_rl_module import A2CRLModule


class A2CTorchRLModule(TorchRLModule, A2CRLModule):
    framework: str = "torch"

    @override(A2CRLModule)
    def setup(self):
        super().setup()

        self._set_inference_only_keys()

    @override(TorchRLModule)
    def get_state(
        self,
        components=None,
        *,
        not_components=None,
        inference_only=False,
        **kwargs,
    ):
        state = super(A2CTorchRLModule, self).get_state(
            components=components, not_components=not_components, **kwargs
        )
        # If this module is not for inference, but the state dict is.
        if not self.config.inference_only and inference_only:
            # Call the local hook to remove or rename the parameters.
            return self._inference_only_get_state_hook(state)
        # Otherwise, the state dict is for checkpointing or saving the model.
        else:
            # Return the state dict as is.
            return state

    def _set_inference_only_keys(self):
        state_dict = self.state_dict()
        self._inference_only_keys = [k for k in state_dict.keys() if "vf" not in k]

    @override(TorchRLModule)
    def _inference_only_get_state_hook(self, state_dict):
        state_dict = self.state_dict()
        return {k: v for k, v in state_dict.items() if k in self._inference_only_keys}

    @override(RLModule)
    def _forward_inference(self, batch):
        # TODO: Support for stateful models
        outputs = self.encoder(batch)
        outputs[Columns.ACTION_DIST_INPUTS] = self.pi(outputs[ENCODER_OUT])

        return outputs

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch)

    @override(RLModule)
    def _forward_train(self, batch):
        outputs = self._forward_inference(batch)

        # Value predictions
        outputs[Columns.VF_PREDS] = self.vf(outputs[ENCODER_OUT]).squeeze(-1)

        # TODO: Input action?
        outputs[Columns.VALUES_BOOTSTRAPPED] = self._compute_bootstrap_values(
            outputs[ENCODER_OUT], batch[Columns.ACTIONS]
        )

        # Bootstrap value predictions
        # TODO: Make this more efficient!
        # next_obs_batch = {Columns.OBS: batch[Columns.NEXT_OBS]}
        # next_outputs = self.encoder(next_obs_batch)
        # outputs[Columns.VALUES_BOOTSTRAPPED] = self.vf(
        #     next_outputs[ENCODER_OUT]
        # ).squeeze(-1)

        return outputs

    def _compute_bootstrap_values(self, encodings, actions):
        actions = self._process_actions_for_model_input(actions)
        model_inputs = torch.cat([encodings.detach(), actions], dim=-1)
        with torch.no_grad():
            next_step_latents = self.dynamics_model(model_inputs)
            bootstrap_values = self.value_model(next_step_latents).squeeze(-1).detach()

        return bootstrap_values

    def unroll_model(self, encodings, actions, n_unroll_steps):
        # TODO: Add an extra hidden layer to each of the model output heads
        reward_preds = []
        vf_preds = []
        actions = self._process_actions_for_model_input(actions)

        latents = encodings
        for t in range(n_unroll_steps):
            model_input = torch.cat([latents, actions[:, t]], dim=-1)
            latents = self.dynamics_model(model_input)
            reward_preds.append(self.reward_model(latents).squeeze(-1))
            vf_preds.append(self.value_model(latents).squeeze(-1))
        reward_preds = torch.stack(reward_preds, dim=1)
        vf_preds = torch.stack(vf_preds, dim=1)

        return reward_preds, vf_preds

    def _process_actions_for_model_input(self, actions):
        return F.one_hot(
            actions.long(), num_classes=self.config.action_space.n
        )  # TODO: Assume discrete actions

    @override(ValueFunctionAPI)
    def compute_values(self, batch):
        vf_encoder_outputs = self.vf_encoder(batch)
        return self.vf(vf_encoder_outputs[ENCODER_OUT]).squeeze(-1)
