from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI

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

        # Bootstrap value predictions
        # TODO: Make this more efficient!
        next_obs_batch = {Columns.OBS: batch[Columns.NEXT_OBS]}
        next_outputs = self.encoder(next_obs_batch)
        outputs[Columns.VALUES_BOOTSTRAPPED] = self.vf(
            next_outputs[ENCODER_OUT]
        ).squeeze(-1)

        return outputs

    @override(ValueFunctionAPI)
    def compute_values(self, batch):
        vf_encoder_outputs = self.vf_encoder(batch)
        return self.vf(vf_encoder_outputs[ENCODER_OUT]).squeeze(-1)
