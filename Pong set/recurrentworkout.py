import numpy as np
import tensorflow as tf

import gym
import matplotlib
from tqdm import tqdm
import mitdeeplearning as mdl

matplotlib.use('TkAgg')

import os

def create_pong_model(shape):
    model = tf.keras.models.Sequential([

        ### convolutional layers with 16 7 filters and 4x4 stride
        tf.keras.layers.Conv2D(filters=16, kernel_size=7, strides=4, padding='same', activation='relu', input_shape=shape),

        ### convolutional layers with 32 5x5 filters and 2x2 stride
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu'),

        ### convolutional layers with 48 3x3 filters and 2x2 stride
        tf.keras.layers.Conv2D(filters=48, kernel_size=3, strides=2, padding='same', activation='relu'),

        ### Flatten the layers
        tf.keras.layers.Flatten(),

        ### Fully connected layer and output
        tf.keras.layers.Dense(units=64, activation='relu'),

        ### Pay attention to the space the agent needs to act in
        tf.keras.layers.Dense(units=6)

    ])
    return model
def create_cart_pole_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=2)
    ])
    return model
def choose_action(model,observation,actions_num):

    ### add batch dimension to the observation
    observation = np.expand_dims(observation, axis=0)

    ### Getting log prob from model
    logits = model.predict(observation)

    ### Softmax to get prob
    prob = tf.nn.softmax(logits).numpy()

    ### Choose action based on prob
    action=np.random.choice(actions_num, size = 1, p = prob.flatten())[0]

    return action
### memory cells for training
class Memory:
    def __init__(self):
        self.clear()
    def clear(self):
        self.observation = []
        self.rewards = []
        self.actions = []
    def remember(self,obesvation,reward,action):
        self.observation.append(obesvation)
        self.rewards.append(reward)
        self.actions.append(action)
### Reward function

def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)
def Reward(rewards,gamma=0.95):
    ret_reward=np.zeros_like(rewards)
    R = 0
    for i in reversed(range(0,len(rewards))):
        R = R*gamma+rewards[i]
        ret_reward[i] = R
    return normalize(ret_reward)

### Defining loss function to compute gradient
def Compute_loss(predict_prob,action_taken_list,rewards):

    ### Compute negative log prop using softmax cross entropy
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_prob, labels=action_taken_list)
    ### multipluing the rewards to amplfy the rewards' effect
    ### reduce means to sum up the loss
    loss = tf.reduce_mean(neg_logprob * rewards)

    return loss
def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        ### Getting model's action
        logits = model(observations)
        ### compute loss of the action
        loss = Compute_loss(logits, actions,  discounted_rewards)
    ### Calculate gradient to train model using gradient tape
    grads = tape.gradient(loss, model.trainable_variables)
    ### Applying backprob to trainable variable
    ### Zip function is to put each gradient with their trainable value
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

### training model
if __name__=="__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    env = gym.make("CartPole-v0")
    env.seed(1)

    ### checking the dimensions of variables needed to train the model
    n_observations = env.observation_space
    print("the dimension of the observed details are: ", n_observations)
    n_actions = env.action_space.n
    print("the number of actions that can be taken are: ", n_observations)

    ### creating models and memory cells
    cartPoleModel = create_cart_pole_model()
    cartPoleMem = Memory()

    ### choosing optimizer and learning rate
    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    ### tracking process using mit deep learning
    smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')

    ### checking if there is an instance of a progress bar and clear it
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

    ### start training
    for iter in range(1):

        ### plot the acquired

        ### reset environment after each run
        observation = env.reset()
        cartPoleMem.clear()

        ### start the game
        while True:
            ### choose action to move
            action=choose_action(cartPoleModel,observation,n_actions)

            ### get results
            next_observation, reward, done, info = env.step(action)

            ### save in memory
            cartPoleMem.remember(next_observation,reward,action)

            ### when the game last too long or the player loses, train the model
            if done:

                ### save point to plot progress
                total_reward=sum(cartPoleMem.rewards)
                smoothed_reward.append(total_reward)

                ### train the model
                train_step(cartPoleModel,optimizer,np.vstack(cartPoleMem.observation),
                           np.array(cartPoleMem.actions),Reward(cartPoleMem.rewards))

                ###reset memory and do over
                cartPoleMem.clear()
                break

            ### continue to next frame
            observation=next_observation

        ### save progress each itter
