from a2c.a2c import A2C
from ray.rllib.connectors.env_to_module.mean_std_filter import MeanStdFilter

algo_config = (
    A2C.get_default_config()
    .training(
        lr=5e-4,
        gamma=0.99,
        train_batch_size_per_learner=2048,
        vf_loss_coeff=1.0,
        entropy_coeff=1e-3,
        bootstrap_horizon=64,
        # grad_clip=10,
    )
    .rl_module(
        model_config_dict={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "swish",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": "swish",
        },
    )
    .environment(env="LunarLander-v2")
    .env_runners(
        num_envs_per_env_runner=32,
        num_env_runners=0,
        env_to_module_connector=lambda x: MeanStdFilter(),
    )
    .reporting(metrics_num_episodes_for_smoothing=50)
)
algo = algo_config.build()
print(algo.learner_group._learner.module)
step = 0
while step < 20_000_000:
    training_results = algo.train()
    algo.save("checkpoint")
    if "module_episode_returns_mean" in training_results["env_runners"]:
        return_mean = training_results["env_runners"]["module_episode_returns_mean"][
            "default_policy"
        ]
        step = training_results["num_env_steps_trained_lifetime"]
        print(f"Step: {step}, Return mean: {return_mean}")
