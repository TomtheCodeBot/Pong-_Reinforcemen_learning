import numpy as np
import gym
import cv2
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
if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    pongMem = recurrentworkout.Memory()
    ### testing the model
    PongModel=recurrentworkout.create_pong_model((80,80,1))
    PongModel.load_weights("trained_pongmodel.h5")
    ### deploy environment
    env = gym.make("Pong-v0")
    env.action_space
    env.seed(2)
    observations = env.reset()

    ### initializing the video output
    score = -1
        ### marker done
    while(score<0):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('pongmodel.avi', fourcc, 35, (160,210))

        done = False
        score=0
        counter = 0
        frame_count=0
        action = 0
        observations = env.reset()
        pongMem.clear()
        prev_frame = pong_image_processing(observations)
        while not done:

            ### get video frame:

            out.write(observations)
            ### getting the diference between frames
            if (frame_count==0):
                curr_frame = pong_image_processing(observations)
                diff_frame = curr_frame - prev_frame
                ### getting action from model
                action = PongModel(np.expand_dims(pong_image_processing(observations), 0)).numpy().argmax()
                prev_frame = curr_frame
                frame_count=0
            else:
                frame_count+=0
            ### getting current state
            observations, reward, done, info = env.step(action)
            score += reward
        print(score)
        out.release()


    print("Successfully saved {} frames into {}!".format(counter, "pongmodel.avi"))


    out2 = cv2.VideoWriter('trained_cartpole.avi', cv2.VideoWriter_fourcc(*'XVID'), 35, (600,400))