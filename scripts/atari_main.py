from drlhp import HumanPreferencesEnvWrapper

import gym
import logging

def main():
    env = gym.make('PongNoFrameskip-v4')
    env.seed(0)

    wrapped_env = HumanPreferencesEnvWrapper(env, segment_length=10, synthetic_prefs=False, 
                                             n_initial_training_steps=10000, env_wrapper_log_level=logging.DEBUG, 
                                             reward_predictor_log_level=logging.DEBUG, pref_interface_log_level=logging.DEBUG)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    n_episodes = 10
    # Fix this
    for episode in range(n_episodes):
        obs = wrapped_env.reset()
        done = False
        while not done:
            action = wrapped_env.action_space.sample()
            next_obs, reward, done, info = wrapped_env.step(action)
            obs = next_obs

        
        logger.info('Episode {} finished'.format(episode))

    wrapped_env.save_prefs()
    wrapped_env.save_reward_predictor()

    wrapped_env.close()

    


if __name__ == '__main__':
    main()