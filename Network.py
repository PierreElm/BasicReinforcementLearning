import torch


# The network class
class Network(torch.nn.Module):

    # Initialisation function
    def __init__(self, input_dimension, output_dimension):
        # Call the parent class
        super(Network, self).__init__()

        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=200)
        self.layer_2 = torch.nn.Linear(in_features=200, out_features=300)
        self.layer_3 = torch.nn.Linear(in_features=300, out_features=200)
        self.output_layer = torch.nn.Linear(in_features=200, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = self.layer_1(input)
        layer_2_output = torch.sigmoid(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, discount_factor=0.0):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a target network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.update_target_network()
        # Define the optimiser which is used when updating the Q-network.
        # The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Discount factor
        self.discount_factor = discount_factor

    # Copy the weights of the Q-network over the target network.
    def update_target_network(self):
        torch.nn.Module.load_state_dict(self.target_network, torch.nn.Module.state_dict(self.q_network))

    # Function that is called whenever we want to train the Q-network.
    # Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        state, action, reward, next_state = transition

        # Max Q_value for next state over target network
        next_state_tensor = torch.tensor(next_state)
        predicted_values = self.target_network.forward(next_state_tensor)
        max_index = torch.unsqueeze(torch.argmax(predicted_values, 1), 1)
        max_q_value = torch.gather(self.q_network.forward(next_state_tensor), 1, max_index)

        # Expected discounted sum of future rewards
        q_value_tensor = torch.tensor(reward) + self.discount_factor * max_q_value

        # Network prediction
        state_tensor = torch.tensor(state)
        network_prediction = self.q_network.forward(state_tensor)
        predicted_q_value = torch.gather(network_prediction, 1, torch.tensor(action))

        # Return the loss
        return torch.nn.MSELoss()(predicted_q_value, q_value_tensor)

