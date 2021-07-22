import numpy as np
import tensorflow as tf

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import mitdeeplearning as mdl
import os

import recurrentworkout

def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)
def Reward(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        ### NEW: Reset the sum if the reward is not 0 (the game has ended!)
        if rewards[t] != 0:
            R = 0
        ### update the total discounted reward as before
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards.astype(np.float32))
def pong_image_processing(frame):
    ### NOTE: this only works specifically with the gym's Pong

    ### cropping the image
    retframe = frame[35:195]

    ### scale it down by a factor of 2
    retframe = retframe[::2, ::2, 0]

    ### reduce background color to 0
    retframe[retframe == 144] = 0
    retframe[retframe == 109] = 0

    ### highlight all the paddles and obstacles
    retframe[retframe != 0] = 1
    return retframe.astype(np.float32).reshape(80, 80, 1)
if __name__=="__main__":
    env = gym.make("Pong-v0")
    env.reset()
     ### looking into the environment
    observation = env.reset()

    print(observation.shape)
    print(pong_image_processing(observation).shape)


    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    env = gym.make("Pong-v0")
    env.seed(1)
    observation = env.reset()

    ### checking the dimensions of variables needed to train the model
    n_observations = env.observation_space
    print("the dimension of the observed details are: ", n_observations)
    n_actions = env.action_space.n
    print("the number of actions that can be taken are: ", n_actions)

    ### create Pong models and memory
    pongModel = recurrentworkout.create_pong_model((80,80,1))
    test_screen = np.expand_dims(pong_image_processing(observation), axis=0)
    print(test_screen.shape)
    pongModel(test_screen)
    pongModel.load_weights("trained_pongmodel.h5")
    pongModel.summary()
    pongMem = recurrentworkout.Memory()

    ### parameters:
    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    iter=1000
    for iter in range(1):

        ### reset environment after each run
        observation = env.reset()
        pongMem.clear()
        prev_frame = pong_image_processing(observation)
        ### start the game
        while True:
            ### getting the diference between frames
            curr_frame = pong_image_processing(observation)
            diff_frame = curr_frame-prev_frame
            ### choosing action
            action = recurrentworkout.choose_action(pongModel, diff_frame, n_actions)
            ### get results
            next_observation, reward, done, info = env.step(action)
            pongMem.remember(diff_frame, reward,action)
            ### when the game last too long or the player loses, train the model
            if done:
                ### save point to plot progress
                total_reward = sum(pongMem.rewards)
                print("itteration {}: {}".format(iter,total_reward))
                ### train the model
                recurrentworkout.train_step(pongModel, optimizer, np.stack(pongMem.observation,0),
                           np.array(pongMem.actions), Reward(pongMem.rewards))

                ###reset memory and do over
                pongMem.clear()
                break

            ### continue to next frame
            observation = next_observation
            pref_frame = curr_frame
        pongModel.save_weights("training_pongmodel.h5")
    pongModel.summary()
    pongModel.save_weights("trained_pongmodel.h5")
    env.close()
   
