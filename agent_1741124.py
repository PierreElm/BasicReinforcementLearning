############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 275
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # Number of step taken in episode
        self.num_steps_taken_episode = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # The deep Q network
        self.dqn = DQN(discount_factor=0.9)
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        # Batch size
        self.batch_size = 100
        # Epsilon
        self.epsilon = 1
        # Delta
        self.delta = 0.00005
        # Evaluate greedy policy
        self.reached_goal = False
        self.reached_goal_in_row = 0
        self.greedy = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken_episode == self.episode_length or self.reached_goal:
            # Reset number of state in episode
            self.num_steps_taken_episode = 0
            # Update Target network at the end of each episode
            self.dqn.update_target_network()
            # We check if we reached the goal
            if self.reached_goal is True:
                self.reached_goal_in_row += 1
            else:
                self.reached_goal_in_row = 0
            # Reset reached_goal for next episode
            self.reached_goal = False
            # If we reached the goal 3 times in a row, we try to apply greedy policy
            if self.reached_goal_in_row >= 3:
                self.greedy = True
            else:
                self.greedy = False
            return True
        else:
            return False

    # Function to get the next action
    def get_next_action(self, state):
        # Choose an action randomly in respect to epsilon greedy.
        if np.random.uniform(0, 1) < self.epsilon and self.greedy is False:
            # The action we chose is biased, since we know the goal is on the right, we prefer go right, top or down.
            discrete_action = np.random.choice([0, 1, 2, 3], 1, p=[0.29, 0.13, 0.29, 0.29])
            # Store the discrete action
            self.action = discrete_action
            # Decrease epsilon
            self.epsilon = max(0, self.epsilon - self.delta)
            # Convert discrete action into continuous action
            action = self.discrete_action_to_continuous(discrete_action)
        # Otherwise, we apply the greedy policy
        else:
            action = self.get_greedy_action(state)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        self.num_steps_taken_episode += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state

        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:  # Move left
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        elif discrete_action == 2:  # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 3:  # Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = self.compute_reward(distance_to_goal)
        # We do nothing if we are applying greedy policy
        if self.greedy is True:
            return
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # We add the transition to the replay buffer
        self.replay_buffer.append_transition(transition)
        # We get a sample of transition
        transition = self.replay_buffer.sample_random_replay_batch(self.batch_size)
        # Train network with this transition
        if transition is not None:
            self.dqn.train_q_network(transition)

    # Function that compute the reward
    def compute_reward(self, distance_to_goal):
        # If we reach the goal
        if distance_to_goal < 0.03:
            self.reached_goal = True
        # If we reach the goal we give a reward of 2.
        if distance_to_goal < 0.03:
            return 2
        # Otherwise we give a negative reward for each step the agent takes.
        return -distance_to_goal

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        state_tensor = torch.unsqueeze(torch.tensor(state), 0)
        # We get the prediction of the network
        predicted_values = self.dqn.q_network.forward(state_tensor).cpu().detach().numpy()
        # We get the best discrete action
        discrete_action = np.argmax(predicted_values)
        # Store the discrete action
        self.action = discrete_action
        # Convert discrete action into continuous action
        action = self.discrete_action_to_continuous(discrete_action)
        return action


# The network class
class Network(torch.nn.Module):

    # Initialisation function
    def __init__(self, input_dimension, output_dimension):
        # Call the parent class
        super(Network, self).__init__()
        # Define the network layers using the decoder model
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=310)
        self.layer_2 = torch.nn.Linear(in_features=310, out_features=245)
        self.layer_3 = torch.nn.Linear(in_features=245, out_features=310)
        # Duel network
        self.value = torch.nn.Linear(in_features=310, out_features=1)
        self.advantage = torch.nn.Linear(in_features=310, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        # Activation functions
        layer_1_output = self.layer_1(input)
        layer_2_output = torch.sigmoid(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        # Duel network
        value = self.value(layer_3_output)
        advantage = self.advantage(layer_3_output)
        output = value + advantage - torch.mean(advantage)

        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, discount_factor=0.9):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a target network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.update_target_network()
        # Define the optimiser which is used when updating the Q-network.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Discount factor
        self.discount_factor = discount_factor

    # Copy the weights of the Q-network over the target network.
    def update_target_network(self):
        torch.nn.Module.load_state_dict(self.target_network, torch.nn.Module.state_dict(self.q_network))

    # Function that is called whenever we want to train the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        state, action, reward, next_state = transition

        # Max Q_value for next state using double deep Q-learning
        next_state_tensor = torch.tensor(next_state)
        # We get the predicted values for the target network and the best action in respect to this network
        predicted_values = self.target_network.forward(next_state_tensor)
        max_index = torch.unsqueeze(torch.argmax(predicted_values, 1), 1)
        # We get the expected value of the best action of the next state from the Q network
        max_q_value = torch.gather(self.q_network.forward(next_state_tensor), 1, max_index)

        # Expected discounted sum of future rewards
        q_value_tensor = (torch.tensor(reward) + self.discount_factor * max_q_value).detach()

        # Network prediction
        state_tensor = torch.tensor(state)
        network_prediction = self.q_network.forward(state_tensor)
        predicted_q_value = torch.gather(network_prediction, 1, torch.tensor(action))

        # Return the loss
        return torch.nn.MSELoss()(predicted_q_value, q_value_tensor)


# Class ReplayBuffer
class ReplayBuffer:

    # Function to initialise a ReplayBuffer object.
    def __init__(self):
        self.collection_deque = collections.deque(maxlen=10000)  # Collection of all transitions

    # Append a transition to the deque.
    def append_transition(self, transition):
        self.collection_deque.append(transition)

    # Sample a random batch of transitions from the replay buffer.
    def sample_random_replay_batch(self, size):
        if len(self.collection_deque) < size:
            return None
        return self.get_transition_batch(np.random.choice(len(self.collection_deque), size, replace=False))

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
