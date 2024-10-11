from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils import deep_update
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LAST_TARGET_UPDATE_TS,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
    NUM_MODULE_STEPS_SAMPLED,
    NUM_MODULE_STEPS_SAMPLED_LIFETIME,
    NUM_MODULE_STEPS_TRAINED,
    NUM_MODULE_STEPS_TRAINED_LIFETIME,
    NUM_TARGET_UPDATES,
    REPLAY_BUFFER_SAMPLE_TIMER,
    REPLAY_BUFFER_UPDATE_PRIOS_TIMER,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TD_ERROR_KEY,
    TIMERS,
)


class A2CConfig(AlgorithmConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or A2C)

        # env_runners()
        # If "auto", set to bootstrap_horizon
        self.rollout_fragment_length = "auto"

        # training()
        self.lr = 3e-4
        self.vf_loss_coeff = 1.0
        self.model_loss_coeff = 1.0
        self.entropy_coeff = 1e-3
        self.train_batch_size_per_learner = 2048
        self.bootstrap_horizon = 64
        self.model_unroll_steps = 5

        # Only works with the new API stack
        self.enable_rl_module_and_learner = True
        self.enable_env_runner_and_connector_v2 = True
        # TODO: What's up with the __prior_exploration_config stuff?

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        vf_loss_coeff=NotProvided,
        model_loss_coeff=NotProvided,
        entropy_coeff=NotProvided,
        bootstrap_horizon=NotProvided,
        model_unroll_steps=NotProvided,
        **kwargs,
    ):
        super().training(**kwargs)

        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if model_loss_coeff is not NotProvided:
            self.model_loss_coeff = model_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if bootstrap_horizon is not NotProvided:
            self.bootstrap_horizon = bootstrap_horizon
        if model_unroll_steps is not NotProvided:
            self.model_unroll_steps = model_unroll_steps

        return self

    @override(AlgorithmConfig)
    def get_rollout_fragment_length(self, worker_index=0):
        if self.rollout_fragment_length == "auto":
            return self.bootstrap_horizon
        else:
            return self.rollout_fragment_length

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self):
        from a2c.torch.a2c_torch_rl_module import A2CTorchRLModule
        from a2c.a2c_catalog import A2CCatalog

        if self.framework_str == "torch":
            return RLModuleSpec(
                module_class=A2CTorchRLModule,
                catalog_class=A2CCatalog,
                model_config_dict=self.model_config,
            )

    @override(AlgorithmConfig)
    def get_default_learner_class(self):
        if self.framework_str == "torch":
            from a2c.torch.a2c_torch_learner import A2CTorchLearner

            return A2CTorchLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported! "
                "Use `config.framework('torch')` instead."
            )

    @override(AlgorithmConfig)
    def build_learner_connector(
        self,
        input_observation_space,
        input_action_space,
        device=None,
    ):
        from ray.rllib.connectors.learner import (
            AddNextObservationsFromEpisodesToTrainBatch,
        )
        from a2c.connectors.add_should_bootstrap_to_batch import (
            AddShouldBootstrapToBatch,
        )

        pipeline = super().build_learner_connector(
            input_observation_space, input_action_space, device
        )
        pipeline.prepend(AddNextObservationsFromEpisodesToTrainBatch())
        pipeline.prepend(AddShouldBootstrapToBatch())

        return pipeline

    @override(AlgorithmConfig)
    def validate(self):
        # TODO: Implement this method
        pass

    @property
    @override(AlgorithmConfig)
    def _model_config_auto_includes(self):
        return super()._model_config_auto_includes | {
            "model_unroll_steps": self.model_unroll_steps
        }


class A2C(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return A2CConfig()

    @override(Algorithm)
    def training_step(self):
        if self.config.enable_env_runner_and_connector_v2:
            return self._training_step_new_api_stack()
        else:
            raise ValueError(
                "Muesli only supports the new API stack! "
                "Use config.api_stack(True, True) to enable it."
            )

    def _training_step_new_api_stack(self):
        with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
            episodes, env_runner_results = synchronous_parallel_sample(
                worker_set=self.env_runner_group,
                max_env_steps=self.config.total_train_batch_size,
                concat=True,
                sample_timeout_s=self.config.sample_timeout_s,
                _uses_new_env_runners=True,
                _return_metrics=True,
            )

        self._log_sampling_metrics(env_runner_results)

        # Learner update step. Just use on-policy data for now and implement
        # a replay buffer later.
        learner_results = self._update_learner_group(episodes)
        self._log_learner_metrics(learner_results)
        self._sync_env_runner_weights(learner_results)

        return self.metrics.reduce()

    def _update_learner_group(self, episodes):
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update_from_episodes(
                episodes=episodes,
                timesteps={
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: (
                        self.metrics.peek(NUM_ENV_STEPS_SAMPLED_LIFETIME)
                    ),
                    NUM_AGENT_STEPS_SAMPLED_LIFETIME: (
                        self.metrics.peek(NUM_AGENT_STEPS_SAMPLED_LIFETIME)
                    ),
                },
            )

        return learner_results

    def _sync_env_runner_weights(self, learner_results):
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            modules_to_update = set(learner_results[0].keys()) - {ALL_MODULES}
            self.env_runner_group.sync_weights(
                from_worker_or_learner_group=self.learner_group,
                policies=modules_to_update,
                global_vars=None,
                inference_only=True,
            )

    def _log_sampling_metrics(self, env_runner_results):
        # Reduce EnvRunner metrics over the n EnvRunners.
        self.metrics.merge_and_log_n_dicts(env_runner_results, key=ENV_RUNNER_RESULTS)

        self.metrics.log_dict(
            self.metrics.peek(
                (ENV_RUNNER_RESULTS, NUM_AGENT_STEPS_SAMPLED), default={}
            ),
            key=NUM_AGENT_STEPS_SAMPLED_LIFETIME,
            reduce="sum",
        )
        self.metrics.log_value(
            NUM_ENV_STEPS_SAMPLED_LIFETIME,
            self.metrics.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED), default=0),
            reduce="sum",
        )
        self.metrics.log_value(
            NUM_EPISODES_LIFETIME,
            self.metrics.peek((ENV_RUNNER_RESULTS, NUM_EPISODES), default=0),
            reduce="sum",
        )
        self.metrics.log_dict(
            self.metrics.peek(
                (ENV_RUNNER_RESULTS, NUM_MODULE_STEPS_SAMPLED),
                default={},
            ),
            key=NUM_MODULE_STEPS_SAMPLED_LIFETIME,
            reduce="sum",
        )

    def _log_learner_metrics(self, learner_results):
        self.metrics.merge_and_log_n_dicts(learner_results, key=LEARNER_RESULTS)
        self.metrics.log_value(
            NUM_ENV_STEPS_TRAINED_LIFETIME,
            self.metrics.peek((LEARNER_RESULTS, ALL_MODULES, NUM_ENV_STEPS_TRAINED)),
            reduce="sum",
        )
        self.metrics.log_dict(
            {
                (LEARNER_RESULTS, mid, NUM_MODULE_STEPS_TRAINED_LIFETIME): (
                    stats[NUM_MODULE_STEPS_TRAINED]
                )
                for mid, stats in self.metrics.peek(LEARNER_RESULTS).items()
                if NUM_MODULE_STEPS_TRAINED in stats
            },
            reduce="sum",
        )

    def _get_current_ts(self):
        if self.config.count_steps_by == "agent_steps":
            current_ts = sum(
                self.metrics.peek(NUM_AGENT_STEPS_SAMPLED_LIFETIME).values()
            )
        else:
            current_ts = self.metrics.peek(NUM_ENV_STEPS_SAMPLED_LIFETIME)

        return current_ts
