import time
import numpy as np

from random_environment import Environment
from agent import Agent


# Main entry point
if __name__ == "__main__":

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    # 1573865367 - 0.63
    # 1573855027 - 77 steps
    # 1573910744 - Impossible
    # 1573912601 - 77 steps
    # 1573913661 - 57 steps
    # 123456 - 73 steps
    # 1573837577 - 84 steps
    # 1573934726 - 78 steps
    # 1573937658 - 87 steps
    # 1573940267 - 71 steps
    # 1573941085 - 67 steps
    # 1573941874 - 62 steps
    # 1573942531 - Can't do it
    # 1573943197 - 59 steps
    # 1573944038 - 0.115723945
    # 1573949022 - 0.47220775
    # 1573949622 - 64 steps
    # 1573987511 - 0.30122116
    # 1573988111 - 71 steps
    # 1573988711 - 65 steps
    # 1573989311 - 64 steps
    # 1573989911 - 0.88026637
    # 1573990511 - 0.5437069
    # 1573991111 - 63 steps
    # 1573991711 - 66 steps
    # 1573992311 - 58 steps
    # 1573993512 - 64 steps
    # 1573994112 - 46 steps
    # 1573994712 - 0.8744835
    # 1573995312 - 60 steps
    # 1573995912 - 61 steps
    # 1573996512 - 0.034
    # 1574034072 - 0.51
    # 1574034693 - 84 steps
    # 1574035313 - 60 steps
    # 1574036393 - 0.657
    # 1574036993 - 0.87
    while True:
        random_seed = int(time.time())
        print('Seed', random_seed)
        np.random.seed(1574036993)

        # Create a random environment
        environment = Environment(magnification=500)

        # Create an agent
        agent = Agent()

        # Get the initial state
        state = environment.init_state

        # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
        start_time = time.time()
        end_time = start_time + 600

        # Train the agent, until the time is up
        while time.time() < end_time:
            # If the action is to start a new episode, then reset the state
            if agent.has_finished_episode():
                state = environment.init_state
            # Get the state and action from the agent
            action = agent.get_next_action(state)
            # Get the next state and the distance to the goal
            next_state, distance_to_goal = environment.step(state, action)
            # Return this to the agent
            agent.set_next_state_and_distance(next_state, distance_to_goal)
            # Set what the new state is
            state = next_state
            # Optionally, show the environment
            if display_on:
                environment.show(state)

        # Test the agent for 100 steps, using its greedy policy
        state = environment.init_state
        has_reached_goal = False
        for step_num in range(100):
            action = agent.get_greedy_action(state)
            next_state, distance_to_goal = environment.step(state, action)
            #print(state)
            # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
            if distance_to_goal < 0.03:
                has_reached_goal = True
                break
            state = next_state

        # Print out the result
        if has_reached_goal:
            print('Reached goal in ' + str(step_num) + ' steps.')
        else:
            print('Did not reach goal. Final distance = ' + str(distance_to_goal))
    #
    # import os
    #
    # duration = 1  # seconds
    # freq = 200  # Hz
    # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
