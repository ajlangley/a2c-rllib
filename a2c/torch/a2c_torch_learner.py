import torch
from torch.nn import functional as F
from torch.optim import AdamW
from ray.rllib.core.learner import Learner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override

from a2c import SHOULD_BOOTSTRAP
from a2c.a2c_learner import (
    A2CLearner,
    LEARNER_RESULTS_POLICY_LOSS_KEY,
    LEARNER_RESULTS_VF_LOSS_KEY,
)


class A2CTorchLearner(A2CLearner, TorchLearner):
    @override(Learner)
    def configure_optimizers_for_module(self, module_id, config=None, hps=None):
        module_params = self.get_parameters(self._module[module_id])
        optimizer = AdamW(module_params, weight_decay=1e-4)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=module_params,
            lr_or_lr_schedule=self.config.lr,
        )

    @override(TorchLearner)
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        if Columns.LOSS_MASK in batch:
            num_valid = torch.sum(batch[Columns.LOSS_MASK])
            possibly_masked_mean = (
                lambda data_: torch.sum(data_[batch[Columns.LOSS_MASK]]) / num_valid
            )
        else:
            possibly_masked_mean = torch.mean

        return_estimates = self._compute_bootstrapped_returns(batch, fwd_out)
        policy_loss = self._compute_policy_loss(
            module_id, return_estimates, batch, fwd_out, possibly_masked_mean
        )
        vf_loss = self._compute_value_loss(
            module_id, fwd_out, return_estimates, possibly_masked_mean
        )

        total_loss = policy_loss + self.config.vf_loss_coeff * vf_loss

        # self._log_loss_metrics(module_id, policy_loss, vf_loss)

        return total_loss

    def _compute_bootstrapped_returns(self, batch, fwd_out):
        rewards = batch[Columns.REWARDS]
        bootstrap_values = self.symexp(fwd_out[Columns.VALUES_BOOTSTRAPPED].detach())
        terminateds = batch[Columns.TERMINATEDS]
        should_boostrap = batch[SHOULD_BOOTSTRAP]

        returns = []
        for i in range(len(rewards)):
            R = 0
            for j in range(i, min(len(rewards), i + self.config.bootstrap_horizon)):
                R += rewards[j] * self.config.gamma ** (j - i)
                if should_boostrap[j]:
                    R += bootstrap_values[j] * self.config.gamma ** (j - i + 1)
                    break
                elif terminateds[j]:
                    break
            returns.append(R)
        returns = torch.tensor(returns, dtype=torch.float32)

        return returns

    def _compute_policy_loss(
        self, module_id, return_estimates, batch, fwd_out, possibly_masked_mean
    ):
        advantages = return_estimates - self.symexp(fwd_out[Columns.VF_PREDS].detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        assert (
            not advantages.requires_grad
            and fwd_out[Columns.ACTION_DIST_INPUTS].requires_grad
            and not batch[Columns.ACTIONS].requires_grad
        )

        # Compute action log probabilities
        action_dist_class_train = (
            self.module[module_id].unwrapped().get_train_action_dist_cls()
        )
        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        actions_logp = curr_action_dist.logp(batch[Columns.ACTIONS])

        policy_losses = (
            -actions_logp * advantages
            - self.config.entropy_coeff * curr_action_dist.entropy()
        )

        return possibly_masked_mean(policy_losses)

    # def _compute_policy_loss(
    #     self, module_id, return_estimates, batch, fwd_out, mean_fn
    # ):
    #     advantages = return_estimates - self.symexp(fwd_out[Columns.VF_PREDS].detach())
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    #     # Compute action dists
    #     action_dist_class_train = (
    #         self.module[module_id].unwrapped().get_train_action_dist_cls()
    #     )
    #     curr_action_dist = action_dist_class_train.from_logits(
    #         fwd_out[Columns.ACTION_DIST_INPUTS]
    #     )

    #     # Likelihood ratio
    #     logp_ratio = torch.exp(
    #         curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
    #     )

    #     # Entropy
    #     curr_entropy = curr_action_dist.entropy()
    #     mean_entropy = mean_fn(curr_entropy)

    #     surrogate_loss = torch.min(
    #         advantages * logp_ratio,
    #         advantages * torch.clamp(logp_ratio, 1 - 0.3, 1 + 0.3),
    #     )

    #     return -mean_fn(surrogate_loss) - self.config.entropy_coeff * mean_entropy

    def _compute_value_loss(
        self, module_id, fwd_out, return_estimates, possibly_masked_mean
    ):
        assert (
            fwd_out[Columns.VF_PREDS].requires_grad
            and not return_estimates.requires_grad
        )
        vf_losses = (
            torch.pow(fwd_out[Columns.VF_PREDS] - self.symlog(return_estimates), 2) / 2
        )
        # TODO: Make clipping param configurable
        # vf_losses = torch.clamp(vf_losses, 0, 10)
        import numpy as np

        r = np.random.choice(np.arange(len(vf_losses)), size=(5,))
        print([self.symexp(fwd_out[Columns.VF_PREDS][ri]).item() for ri in r])
        print([return_estimates[ri].item() for ri in r])
        print(vf_losses.mean(), vf_losses.requires_grad)

        return possibly_masked_mean(vf_losses)

    def symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def symexp(self, x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)