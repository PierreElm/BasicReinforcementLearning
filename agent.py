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
from Network import DQN
from replayBuffer import ReplayBuffer


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 350
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # The deep Q network
        self.dqn = DQN(0.9)
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        # Batch size
        self.batch_size = 450
        # Epsilon
        self.epsilon = 1
        # Delta
        self.delta = 0.00008
        self.last_distance = None
        self.last_state = None
        # Random on episode
        self.epsilon_episode = 0
        self.delta_episode = 0.1

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            print(self.epsilon)
            print(self.last_distance)
            self.epsilon_episode = 0.0
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state, discrete_action=None):
        # Here, the action is random, but you can change this
        #action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        # Choose an action with e-greedy policy
        if discrete_action is None:
            if np.random.uniform(0, 1) <= self.epsilon_episode and self.epsilon < 0:
                discrete_action = np.random.randint(0, 4, 1)[0]
                # Store the discrete action
                self.action = discrete_action
                # Decrease epsilon
                self.epsilon_episode = max(0, self.epsilon_episode - self.delta_episode)
                self.epsilon = max(0, self.epsilon - self.delta)
                # Convert discrete action into continuous action
                action = self.discrete_action_to_continuous(discrete_action)
                if (self.last_state == self.state).all():
                    self.epsilon_episode = 1
            elif np.random.uniform(0, 1) <= self.epsilon or self.state is None:
                discrete_action = np.random.randint(0, 4, 1)[0]
                # Store the discrete action
                self.action = discrete_action
                # Decrease epsilon
                self.epsilon = max(0, self.epsilon - self.delta)
                # Convert discrete action into continuous action
                action = self.discrete_action_to_continuous(discrete_action)
            elif (self.last_state == self.state).all():
                self.epsilon_episode = 0.8
                discrete_action = np.random.randint(0, 4, 1)[0]
                # Store the discrete action
                self.action = discrete_action
                # Convert discrete action into continuous action
                action = self.discrete_action_to_continuous(discrete_action)
            else:
                action = self.get_greedy_action(self.state)

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.last_state = self.state
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
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # We add the transition to the replay buffer
        self.replay_buffer.append_transition(transition)
        # We get a sample of transition
        transition = self.replay_buffer.sample_random_replay_batch(self.batch_size)
        # Train network with this transition
        if transition is not None:
            loss, weight = self.dqn.train_q_network(transition)
        # Every 25 steps, we update the target network
        if self.num_steps_taken % 150 == 0:
            self.dqn.update_target_network()

    # Function that compute the reward
    def compute_reward(self, distance_to_goal):
        self.last_distance = distance_to_goal
        if distance_to_goal < 0.1:
            return 2 - distance_to_goal
        return 1 - distance_to_goal

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        state_tensor = torch.unsqueeze(torch.tensor(state), 0)
        predicted_values = self.dqn.q_network.forward(state_tensor).cpu().detach().numpy()
        discrete_action = np.argmax(predicted_values)
        # Store the discrete action
        self.action = discrete_action
        # Convert discrete action into continuous action
        action = self.discrete_action_to_continuous(discrete_action)
        return action
