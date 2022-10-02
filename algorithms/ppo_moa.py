from ray.rllib.agents.ppo import PPOTrainer

# Import PPOLoss from ray.rllib.agents.ppo.ppo_tf_policy.py

from algorithms.old_ppo_loss import PPOLoss
from ray.rllib.agents.ppo.ppo import ppo_tf_policy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import EntropyCoeffSchedule, LearningRateSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf
from algorithms.common_funcs_moa import (
    EXTRINSIC_REWARD,
    SOCIAL_INFLUENCE_REWARD,
    MOAResetConfigMixin,
    build_model,
    get_moa_mixins,
    moa_fetches,
    moa_postprocess_trajectory,
    setup_moa_loss,
    setup_moa_mixins,
    validate_moa_config,
)

tf1, tf, version = try_import_tf()

POLICY_SCOPE = "func"

def clip_gradients(policy, optimizer, loss):
    variables = policy.model.trainable_variables()
    if policy.config["grad_clip"] is not None:
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads,
                                                 policy.config["grad_clip"])
        clipped_grads = list(zip(policy.grads, variables))
        return clipped_grads
    else:
        return optimizer.compute_gradients(loss, variables)

def setup_config(policy, obs_space, action_space, config):
    # auto set the model option for layer sharing
    config["model"]["vf_share_layers"] = config["vf_share_layers"]

def loss_with_moa(policy, model, dist_class, train_batch):
    """
    Calculate PPO loss with MOA loss
    :return: Combined PPO+MOA loss
    """
    # you need to override this bit to pull out the right bits from train_batch
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    moa_loss = setup_moa_loss(logits, policy, train_batch)
    policy.moa_loss = moa_loss.total_loss
    
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)
        
        policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        policy.entropy_coeff,
        policy.config["clip_param"],
        policy.config["vf_clip_param"],
        policy.config["vf_loss_coeff"],
        policy.config["use_gae"],
    )
    policy.loss_obj.loss += moa_loss.total_loss
    return policy.loss_obj.loss

def extra_moa_fetches(policy):
    """
    Adds value function, logits, moa predictions to experience train_batches.
    :return: Updated fetches
    """
    ppo_fetches = ppo_tf_policy.vf_preds_fetches(policy)
    ppo_fetches.update(moa_fetches(policy))
    return ppo_fetches

def extra_moa_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined PPO+MOA stats
    """
    base_stats = ppo_tf_policy.kl_and_loss_stats(policy, train_batch)
    base_stats = {
        **base_stats,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "cur_influence_reward_weight": tf.cast(
            policy.cur_influence_reward_weight_tensor, tf.float32
        ),
        SOCIAL_INFLUENCE_REWARD: train_batch[SOCIAL_INFLUENCE_REWARD],
        EXTRINSIC_REWARD: train_batch[EXTRINSIC_REWARD],
        "moa_loss": policy.moa_loss,
    }

    return base_stats


def postprocess_ppo_moa(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Add the influence reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = moa_postprocess_trajectory(policy, sample_batch)
    batch = ppo_tf_policy.postprocess_ppo_gae(policy, batch)
    return batch


def setup_ppo_moa_mixins(policy, obs_space, action_space, config):
    """
    Calls init on all PPO+MOA mixins in the policy
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_moa_mixins(policy, obs_space, action_space, config)

def validate_ppo_moa_config(config):
    """
    Validates the PPO+MOA config
    :param config: The config to validate
    """
    validate_moa_config(config)
    validate_config(config)
    
def build_ppo_moa_trainer(moa_config):
    """
    Creates a MOA+PPO policy class, then creates a trainer with this policy.
    :param moa_config: The configuration dictionary.
    :return: A new MOA+PPO trainer.
    """
    tf.keras.backend.set_floatx("float32")
    trainer_name = "MOAPPOTrainer"

    moa_ppo_policy = build_tf_policy(
        name="MOAPPOTFPolicy",
        get_default_config=lambda: moa_config,
        loss_fn=loss_with_moa,
        make_model=build_model,
        stats_fn=extra_moa_stats,
        extra_action_fetches_fn=extra_moa_fetches,
        postprocess_fn=postprocess_ppo_moa,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_ppo_moa_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin]
        + get_moa_mixins(),
    )
    
    moa_ppo_trainer = PPOTrainer(config=moa_config, policy_cls=moa_ppo_policy, name=trainer_name)
    
    return moa_ppo_trainer