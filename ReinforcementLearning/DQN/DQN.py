import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np


class DeepQNetwork():
  """Discrete Action Space Deep Q Network implementation."""
  def __init__(self, state_dimensions, action_dimensions,
               steps_before_copy=500):
    """Creates a DQN.
    Args:
      state_dimensions: The number of dimensions of the state input.
      action_dimensions: The number of dimensions of the action space.
      steps_before_copy: The number of training steps that happen before 
        copying the source network to the target network.
    """
    self.state_dimensions = state_dimensions
    self.action_dimensions = action_dimensions
    self.steps_before_copy = steps_before_copy
    # State 1 is the current state.
    self.state_1 = tf.placeholder(tf.float32, (None, state_dimensions))
    # Action that is to be taken
    self.action = tf.placeholder(tf.float32, (None, action_dimensions))
    # State 2 is the state after a chosen action is taken.
    self.state_2 = tf.placeholder(tf.float32, (None, state_dimensions))
    self.reward = tf.placeholder(tf.float32)
    # These represent the ouput of the source and target networks.
    self.q_source = self.create_q_network(self.state_1, "q_source")
    self.q_target = self.create_q_network(self.state_2, "q_target")
    self.copy_operation = self.create_copy_operation()
    self.loss_done = slim.losses.sum_of_squares()
    self.loss_not_done = slim.losses.sum_of_squares()

  def create_copy_operation(self):
    """Creates TF op that copys all of the variables from source to target."""
    scope_name = tf.get_variable_scope().name
    source_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope_name + "/q_source")
    target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope_name + "/q_target")
    if len(source_variables) != len(target_variables):
      raise ValueError("Source variables and target "
                       "variables are not the same size.")
    copy_operations = []
    for target, source in zip(target_variables, source_variables):
      copy_operations.append(tf.assign(target, source))
    return tf.group(*copy_operations, name="copy_souce_to_target")

  def train(state1, action, reward, state2, session):
    if state2 is None:
      # This means that the episode is over
      pass


  def create_q_network(self, input_tensor, scope):
    """Creates a fully connected network based on dimesions of self.
    Args:
      input_tensor: The tensor from the state.
      scope: A string for the scope you want the tensor to be under.
    Returns:
      output: Output of the network of the network.
    """
    with tf.variable_scope(scope):
      # Pass this through a few fully connected layers.
      # Slim is magic. I highly recommend reading up the API if 
      # you have the chance. Our entire network can be built in a 
      # single line of code.
      output = slim.stack(input_tesnor, slim.fully_connected,
          [32, 64, 64, self.action_dimensions])
      return output
