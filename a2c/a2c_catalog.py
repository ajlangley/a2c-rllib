from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
    FreeLogStdMLPHeadConfig,
    MLPHeadConfig,
    RecurrentEncoderConfig,
)
from ray.rllib.utils.annotations import override


class A2CCatalog(Catalog):
    @override(Catalog)
    def __init__(
        self,
        observation_space,
        action_space,
        model_config_dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
        self.output_head_hiddens = self._model_config_dict["post_fcnet_hiddens"]
        self.output_head_activation = self._model_config_dict["post_fcnet_activation"]

    def build_pi_head(self, framework):
        return self._get_head_config(is_pi_head=True, framework=framework).build(
            framework=framework
        )

    def build_single_output_head(self, framework):
        return self._get_head_config(output_dim=1).build(framework=framework)

    def build_dynamics_model(self, framework):
        return MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self._model_config_dict.get(
                "dyn_model_hiddens", (self.latent_dims[-1],) * 2
            ),
            hidden_layer_activation=self._model_config_dict.get(
                "dyn_model_activation", self.output_head_activation
            ),
            output_layer_activation="linear",
        ).build(framework=framework)

    def _get_head_config(
        self, input_dim=None, output_dim=None, is_pi_head=False, framework="torch"
    ):
        head_config_class = MLPHeadConfig
        if is_pi_head:
            action_distribution_cls = self.get_action_dist_cls(framework=framework)
            if self._model_config_dict["free_log_std"]:
                _check_if_diag_gaussian(
                    action_distribution_cls=action_distribution_cls, framework=framework
                )
            output_dim = action_distribution_cls.required_input_dim(
                space=self.action_space, model_config=self._model_config_dict
            )
            head_config_class = (
                FreeLogStdMLPHeadConfig
                if self._model_config_dict["free_log_std"]
                else MLPHeadConfig
            )

        return head_config_class(
            input_dims=(input_dim,) if input_dim is not None else self.latent_dims,
            hidden_layer_dims=self.output_head_hiddens,
            hidden_layer_activation=self.output_head_activation,
            output_layer_activation="linear",
            output_layer_dim=output_dim,
        )
