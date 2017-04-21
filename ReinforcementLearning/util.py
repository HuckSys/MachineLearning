class ReplayBuffer():
  """A very basic replay buffer implementation."""
  def __init__(self, size=2000):
    self.size = size