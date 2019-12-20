import gym
import random, os
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import scipy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

example_folderPath = 'D:\\mine\\Work\\OpenAI\\cart_pole'

## Learing rate for training the model
LR = 1e-3
## Create the OpenAI gym environment
env = gym.make("CartPole-v0")
## Make sure everything is reset at the beginning
env.reset()
## Number of steps per episode
goal_steps = 500
## Desired goal value to be used for filtering the outliers
score_threshold = 50
## Number of episodes for the training
training_episodes_num = 10000

def generate_training_data():
    # Store the training data in terms of[OBS, MOVES] per each step
    training_data = []
    # Record all scores per each episode for doing some analysis later:
    scores = []
    # Record only the filtered scores that met our threshold:
    accepted_scores = []
    # Here we generate our training data by iterating through however many episodes we want:
    for _ in range(training_episodes_num):      ## EPISODES PARENT LOOP
        # Initial score for the episode
        score = 0
        # Store all observations and the corresponding actions per each episode
        episode_memory = []
        # Store the previous observation that we saw, because we store in the memory the current action with the previous observation
        # [prev_observ, action] represents one item of our training data
        prev_observation = []
        # Here we iterate over number of steps per each episode, to generate the training data
        for _ in range(goal_steps):            
            # Show the cart-pole window
            # env.render()
            # Pick an initial action at the beginning of each step.
            action = random.randrange(0,2)
            # Apply this action on the Agent and take the corresponding observation, the termination state, and reward value.
            observation, reward, done, info = env.step(action)
            
            # Check if there is at least one previous observation, store the resulted action based on it.
            if len(prev_observation) > 0 :
                # Here we store the previous observation and the resulted action from it.
                episode_memory.append([prev_observation, action])
            # The current observation at this iteration will be the previous observation for the next iteration.
            prev_observation = observation
            # Accumlate the episode score value by adding this step's reward value.
            score+=reward
            # Here is the cart-pole reached a forbidden situation, so this episode is finished now and let's try a new episode after resetting
            if done: break

        # Here we're do some filtering on the score data, 
        # IF our score is higher than our threshold, we'd like to save every move we made
        # Here all we're doing is reinforcing the score, we're trying only to inforce the machine 
        # to learn with those good situations
        if score >= score_threshold:
            # Storing the filtered scores to do some analysis on this training data.
            accepted_scores.append(score)
            # Here we store the training data items from the episode memory
            for data in episode_memory:
                # data is a training item [observation, action]
                # Here we determine the output value for this accepted score
                if data[1] == 1:        # Here the action is to go right for example
                    output = [0,1]
                elif data[1] == 0:      # Here the action is to go left for example.
                    output = [1,0]
                    
                # saving our training data [prev_observation, outputAction]
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    
    # Update our game_memory training data to be used later if we need
    new_training_data = np.array(training_data)
    old_training_data = np.load(example_folderPath + '\\data\\saved.npy') \
    if os.path.isfile(example_folderPath + '\\data\\saved.npy') else [] #get data if exist
    np.save(example_folderPath + '\\data\\saved.npy',np.append(old_training_data, new_training_data))
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data


def neural_network_model(input_size):
    """
    shape: list of `int`. An array or tuple representing input data shape.
            It is required if no placeholder is provided. First element should
            be 'None' (representing batch size), if not provided, it will be
            added automatically.
    """
    # input_size is the size of the training data array.
    network = input_data(shape=[None, input_size, 1], name='input')

    # Hidden layers(Fully connected layers with 20% dropout and relu activation function)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    # network = fully_connected(network, 512, activation='relu')
    # network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    # Output layer(targets layer) with softmax activation and 2 output nodes for the 2 actions
    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    # Training data structure is [[[prev_observ], output], ... n training example]
    # And we need it in (n, 1) format to be compatible with the input layer neurals
    # So we'll use reshape function to reach this format, by creating a 3D matirx(x,y,z) to be Array of 2D matrices
    # x: rows number which's number of all training examples
    # y: row number of each sub-matrix, which's number of observations per each matrix
    # z: columns number of each sub-matrix, which's number of number of examples to be taken
    """
    reshape(-1,...) : It simply means that it is an unknown dimension and we want numpy to figure it out. 
    And numpy will figure this by looking at the 'length of the array and remaining dimensions'
    """
    
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    print(X)
    y = [i[1] for i in training_data]
    if not model:   # if we don't have a pre-trained model, create a new one.
        model = neural_network_model(input_size = len(X[0]))    # Takes number of the training examples
    """
    one epoch = one forward pass and one backward pass of all the training examples
    batch size = the number of training examples in one forward/backward pass. 
    The higher the batch size, the more memory space you'll need.
    number of iterations = number of passes, each pass using [batch size] number of exam

    WHY several epochs:
    Neural networks are typically trained using an iterative optimization method (most of the time, gradient descent), 
    which often needs to perform several passes on the training set to obtain good results.
    """
    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='cart_pole_learning')
    return model

training_data = generate_training_data()
model = train_model(training_data)


scores = []
choices = []
for each_game in range(150):
    score = 0
    episode_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        # env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        choices.append(action)
                
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        episode_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_threshold)
model.save("cart_pole\\models\\" + str(sum(scores)/len(scores)))