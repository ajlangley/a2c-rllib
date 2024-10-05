from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns


class A2CRLModule(RLModule):
    @override(RLModule)
    def setup(self):
        catalog = self.config.get_catalog()

        self.encoder = catalog.build_encoder(framework=self.framework)
        # Build heads
        self.pi = catalog.build_pi_head(framework=self.framework)
        if not self.config.inference_only:
            self.vf = catalog.build_vf_head(framework=self.framework)

        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)

    @override(RLModule)
    def get_exploration_action_dist_cls(self):
        return self.action_dist_cls

    @override(RLModule)
    def get_inference_action_dist_cls(self):
        return self.action_dist_cls

    @override(RLModule)
    def get_train_action_dist_cls(self):
        return self.action_dist_cls

    @override(RLModule)
    def get_initial_state(self):
        # TODO: Support for stateful models
        return {}

    @override(RLModule)
    def input_specs_inference(self):
        return [Columns.OBS]

    @override(RLModule)
    def input_specs_exploration(self):
        return self.input_specs_inference()

    @override(RLModule)
    def input_specs_train(self):
        return [
            Columns.OBS,
            Columns.ACTIONS,
            Columns.NEXT_OBS,
        ]

    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_exploration(self):
        return self.output_specs_inference()

    @override(RLModule)
    def output_specs_train(self):
        return [
            Columns.ACTION_DIST_INPUTS,
            Columns.VF_PREDS,
            Columns.VALUES_BOOTSTRAPPED,
        ]
