import collections
import numpy as np


# Class ReplayBuffer
class ReplayBuffer:

    # Function to initialise a ReplayBuffer object.
    def __init__(self):
        self.collection_deque = collections.deque(maxlen=1000000)

    # Append a transition to the deque.
    def append_transition(self, transition):
        self.collection_deque.append(transition)

    # Sample a random batch of transitions from the replay buffer.
    def sample_random_replay_batch(self, size):
        if len(self.collection_deque) < size:
            return None
        return self.get_transition_batch(np.random.choice(len(self.collection_deque), size))

    # Return a batch of transition
    def get_transition_batch(self, indexes):
        batch_input = []
        batch_reward = []
        batch_direction = []
        batch_next_state = []

        # Create the batch transition
        for index in indexes:
            batch_reward.append([self.collection_deque[index][2]])
            batch_input.append(self.collection_deque[index][0])
            batch_direction.append([int(self.collection_deque[index][1])])
            batch_next_state.append(self.collection_deque[index][3])

        return batch_input, batch_direction, batch_reward, batch_next_state
