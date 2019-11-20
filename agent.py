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
        # Last distance, this is used to stop the episode earlier if we reach the goal
        self.last_distance = None
        # The last state we were in, this is used to take random action if we go toward a wall
        self.last_state = None
        # Evaluate greedy policy
        self.random = False
        self.step_to_goal = 300
        self.reached_goal = False
        self.reached_goal_in_row = 0
        self.greedy = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken_episode == self.episode_length or self.reached_goal:
            # Reset number of state in episode
            self.num_steps_taken_episode = 0
            # Update Target network
            self.dqn.update_target_network()

            # We check if we reached the goal
            if self.reached_goal is True:
                self.reached_goal_in_row += 1
            else:
                self.reached_goal_in_row = 0
            self.reached_goal = False

            # If we reached the goal 10 times in a row, we try to apply greedy policy
            if self.reached_goal_in_row >= 3:
                self.greedy = False
            else:
                self.greedy = False
            print(self.epsilon)
            return True
        else:
            return False

    # Function to get the next action
    def get_next_action(self, state):
        # Choose an action randomly
        if np.random.uniform(0, 1) < self.epsilon and self.greedy is False:
            # The action we chose is biased, since we know the goal is on the right, we prefer go right, top or down.
            discrete_action = np.random.randint(0, 4, 1)[0]
            # Store the discrete action
            self.action = discrete_action
            # Decrease epsilon
            self.epsilon = max(0, self.epsilon - self.delta)
            # Convert discrete action into continuous action
            action = self.discrete_action_to_continuous(discrete_action)
            self.random = True

        # Choose random action if the agent stayed still
        elif (self.last_state == self.state).all() and self.greedy is False and False:
            discrete_action = np.random.randint(0, 4, 1)[0]
            self.epsilon = max(0, self.epsilon - self.delta)
            # Store the discrete action
            self.action = discrete_action
            # Convert discrete action into continuous action
            action = self.discrete_action_to_continuous(discrete_action)
            self.random = True

        # Otherwise, we apply the greedy policy
        else:
            action = self.get_greedy_action(self.state)

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        self.num_steps_taken_episode += 1
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

        self.last_distance = distance_to_goal
        # If we reach an area that is close to the goal, we increase a bit the reward to give more feedback to the agent
        if distance_to_goal < 0.03:
            return 5
        # Otherwise we use the original reward that seems to perform well on different environment
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
