from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.annotations import override

from a2c import SHOULD_BOOTSTRAP


class AddShouldBootstrapToBatch(ConnectorV2):
    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module,
        data,
        episodes,
        explore=None,
        shared_data=None,
        **kwargs,
    ):
        for sa_episode in self.single_agent_episode_iterator(
            episodes,
            agents_that_stepped_only=False,
        ):
            should_boostrap = not sa_episode.is_terminated
            self.add_n_batch_items(
                data,
                SHOULD_BOOTSTRAP,
                items_to_add=[False] * (len(sa_episode) - 1) + [should_boostrap],
                num_items=len(sa_episode),
                single_agent_episode=sa_episode,
            )

        return data
