import collections
import numpy as np


# Class ReplayBuffer
class ReplayBuffer:

    # Function to initialise a ReplayBuffer object.
    def __init__(self):
        self.collection_deque = collections.deque(maxlen=15000)  # Collection of all transitions
        self.weight_deque = collections.deque(maxlen=15000)  # Collection of weight of transitions
        self.last_indexes = []  # Last indexes used in mini batch
        self.probability = None

        # Metrics used for prioritised replay
        self.epsilon = 0.001
        self.alpha = 2
        self.max_weight = 0
        self.weight_selected = 0
        self.total_weight = 0

    # Append a transition to the deque.
    def append_transition(self, transition):
        self.collection_deque.append(transition)
        self.weight_deque.append(self.max_weight)

    def sample_prioritised_replay_batch(self, size):
        if len(self.collection_deque) < size:
            return None
        elif self.probability is None:
            return self.sample_random_replay_batch(size)
        else:
            self.last_indexes = np.argpartition(self.probability, -size)[-size:]

            for index in self.last_indexes:
                self.weight_selected += self.weight_deque[index]
            return self.get_transition_batch(self.last_indexes)

    def update_weights(self, magnitude):
        old_max_weight = self.max_weight
        new_weight_selected = 0
        i = 0
        for index in self.last_indexes:
            weight = np.abs(magnitude[i]) + self.epsilon
            self.max_weight = max(self.max_weight, weight)
            self.weight_deque[index] = weight
            new_weight_selected += weight**self.alpha
            i += 1
        self.total_weight = old_max_weight + (new_weight_selected - self.weight_selected)

        self.probability = np.empty([len(self.collection_deque)])
        for i in range(0, len(self.collection_deque)):
            self.probability[i] = self.weight_deque[i]**self.alpha / self.total_weight

    # Sample a random batch of transitions from the replay buffer.
    def sample_random_replay_batch(self, size):
        if len(self.collection_deque) < size:
            return None
        self.last_indexes = np.random.choice(len(self.collection_deque), size)
        return self.get_transition_batch(self.last_indexes)

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
