import agent
import tensorflow as tf
import cv2
from tf_agents.environments import tf_py_environment

eval_env = tf_py_environment.TFPyEnvironment(agent.Game(limit=False))

saved_policy = tf.compat.v2.saved_model.load('policies/epoch_run/policy_20000')
time_step = eval_env.reset()
while True:
    policy_step = saved_policy.action(time_step)
    time_step = eval_env.step(policy_step.action)
    cv2.imshow('frame', cv2.resize(eval_env.render()[0].numpy(), (240, 400), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(33)
    if time_step.is_last():
        time_step = eval_env.reset()
