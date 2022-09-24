"""Note: Keep in sync with changes to VTraceTFPolicy."""

from __future__ import absolute_import, division, print_function

from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_gae_for_sample_batch
from ray.rllib.policy.sample_batch import SampleBatch

# from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.torch_mixins import LearningRateSchedule, ValueNetworkMixin

# from ray.rllib.utils import try_import_torch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.torch_utils import explained_variance

from algorithms.common_funcs_moa import (
    EXTRINSIC_REWARD,
    SOCIAL_INFLUENCE_REWARD,
    get_moa_mixins,
    moa_fetches,
    moa_postprocess_trajectory,
    setup_moa_loss,
    setup_moa_mixins,
)

# torch, nn = try_import_torch()
tf1, tf, version = try_import_tf()


class A3CLoss(object):
    def __init__(
        self,
        action_dist,
        actions,
        advantages,
        v_target,
        vf,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
    ):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = self.pi_loss + self.vf_loss * vf_loss_coeff - self.entropy * entropy_coeff


def postprocess_a3c_moa(policy, sample_batch, other_agent_batches=None, episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    batch = moa_postprocess_trajectory(policy, sample_batch)
    batch = compute_gae_for_sample_batch(policy, batch)
    return batch


def actor_critic_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.loss = A3CLoss(
        action_dist,
        train_batch[SampleBatch.ACTIONS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[Postprocessing.VALUE_TARGETS],
        model.value_function(),
        policy.config["vf_loss_coeff"],
        policy.config["entropy_coeff"],
    )

    moa_loss = setup_moa_loss(logits, policy, train_batch)
    policy.loss.total_loss += moa_loss.total_loss

    # store this for future statistics
    policy.moa_loss = moa_loss.total_loss

    return policy.loss.total_loss


def add_value_function_fetch(policy):
    fetch = {SampleBatch.VF_PREDS: policy.model.value_function()}
    fetch.update(moa_fetches(policy))
    return fetch


def stats(policy, train_batch):
    base_stats = {
        "cur_lr": policy.cur_lr,
        "policy_loss": policy.loss.pi_loss,
        "policy_entropy": policy.loss.entropy,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "vf_loss": policy.loss.vf_loss,
        "cur_influence_reward_weight": tf.cast(
            policy.cur_influence_reward_weight_tensor, tf.float32
        ),
        SOCIAL_INFLUENCE_REWARD: train_batch[SOCIAL_INFLUENCE_REWARD],
        EXTRINSIC_REWARD: train_batch[EXTRINSIC_REWARD],
        "moa_loss": policy.moa_loss,
    }
    return base_stats


def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.global_norm(grads),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy.model.value_function()
        ),
    }


def clip_gradients(policy, optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(loss, policy.model.trainable_variables())
    grads = [g for (g, v) in grads_and_vars]
    grads, _ = tf.clip_by_global_norm(grads, policy.config["grad_clip"])
    clipped_grads = list(zip(grads, policy.model.trainable_variables()))
    return clipped_grads


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_moa_mixins(policy, obs_space, action_space, config)


def build_a3c_moa_trainer(moa_config):
    trainer_name = "MOAA3CTrainer"
    moa_config["use_gae"] = False

    # recreate the above in ray 2.0 with pytorch
    a3c_policy = build_tf_policy(
        name="A3CAuxTorchPolicy",
        get_default_config=lambda: moa_config,
        loss_fn=actor_critic_loss,
        stats_fn=stats,
        grad_stats_fn=grad_stats,
        gradients_fn=clip_gradients,
        postprocess_fn=postprocess_a3c_moa,
        extra_action_fetches_fn=add_value_function_fetch,
        before_loss_init=setup_mixins,
        mixins=[ValueNetworkMixin, LearningRateSchedule] + get_moa_mixins(),
    )

    trainer = A3CTrainer.with_updates(
        name=trainer_name,
        default_policy=a3c_policy,
        default_config=moa_config,
    )

    return trainer
