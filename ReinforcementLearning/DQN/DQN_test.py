import unittest
from ReinforcementLearning.DQN import DQN
class TestDQN(unittest.TestCase):
  def testSanityCheck(self):
    """Verify that IO shapes are correct"""
    dqn = DQN.DeepQNetwork(
        state_dimensions=4,
        action_dimensions=3)



if __name__ == '__main__':
  unittest.main()