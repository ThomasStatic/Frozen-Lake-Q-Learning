import gym
import random
import numpy as np

renderTheGame = input("Do you want to render the environment?")
if(renderTheGame.lower() == "yes"):
    # create the training environment
    environment = gym.make("FrozenLake-v1", desc = None, map_name="4x4", is_slippery=True, render_mode= "human")
else:
    environment = gym.make("FrozenLake-v1", desc = None, map_name="4x4", is_slippery=True)


environment.reset()

# Comment/uncomment this line to see the actual map when training the model
environment.render() 

qRows = environment.observation_space.n
qColumns = environment.action_space.n

qTable = np.zeros((qRows, qColumns))
print(qTable)
print(f"States = {qRows}, Actions = {qColumns}")

# learning rate
learnRate = 0.8

# discount factor
discRate = 0.9

# likelihood the agent will explore something new
# bumping up episodes substantially so increasing this
epsi = 0.92

# decay rate 
decRate = 0.01

numEpisodes = 10000
numSteps = 100

# Q-Learning Algorithm
for episode in range(numEpisodes):

    # Termination conditions: player moves into hole or reaches goal
    terminated = False

    # Truncation condition: length of episode exceeds the 100 step limit
    truncated = False

    # Start back in the top left corner everytime
    state = environment.reset()[0]

    # This loop runs for 100 steps or until termination/truncation
    for step in range(numSteps):

        if random.uniform(0, 1) < epsi:
            action = environment.action_space.sample()

        else:
            action = np.argmax(qTable[state, :])

        # print(env.step(action))
        newState, reward, truncated, terminated, _ = environment.step(action)

        # 1 is going down and 2 is going to the right, so we are rewarding moving down to the right more
        if(action == 1 or action == 2):
            reward = reward+1
        

        qTable[state, action] = qTable[state, action] + learnRate * \
                                (reward + discRate * np.max(qTable[newState, :]) - qTable[state, action])

        print(f"Step: {step} of episode {episode}")
        state = newState
        # print(type(state))

        if truncated or terminated:
            break

    epsi = np.exp(-decRate * episode)
    print(f"Epsilon: {epsi}")


truncated = False
terminated = False
score = 0
step = 0
state = environment.reset()[0]

game = input("Do you want to watch the  AI play: ")

if game.lower() == "yes":
    environment = gym.make("FrozenLake-v1", desc = None, map_name="4x4", is_slippery=True, render_mode= "human")
    state = environment.reset()[0]
    #Actually Playing
    while not(truncated or terminated):
        print(f"Step {step}")
        action = np.argmax(qTable[state, :])
        newState, reward, truncated, terminated, _ = environment.step(action)
        score += reward
        environment.render()
        print(f"Score = {score}")
        state = newState
        step += 1

    environment.close()