'''
    Training limited holdem agent with DQN RL
'''

import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent

from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1
from rlcard.agents import DQNAgent

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args, random_prob_list, curriculum): 

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        if args.load_checkpoint_path != "":
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[128, 256, 128,64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
        
    '''
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[64, 64],
            q_mlp_layers=[64, 64],
            device=device,
            save_path=args.log_dir,
            save_every=args.save_every
        )
    '''

    agents = [agent]
 

    for random_prob in random_prob_list:

        for _ in range(1, env.num_players):

            if(curriculum == False):
                agents.append(RandomAgent(num_actions=env.num_actions))

            else:
                agents.append(LimitholdemRuleAgentV1(random_prob=random_prob))

            #Load a pre-trained DQN-random model
            #pre_trained_model_path = 'experiments/limit_holdem_dqn_vs_random__ver0/checkpoint_dqn.pt'
            #agents.append(DQNAgent.from_checkpoint(checkpoint=torch.load(pre_trained_model_path))) 


        env.set_agents(agents)

        # Start training
        with Logger(args.log_dir) as logger:
            for episode in range(args.num_episodes):

                if args.algorithm == 'nfsp':
                    agents[0].sample_episode_policy()

                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                # Here, we assume that DQN always plays the first position
                # and the other players play randomly (if any)
                for ts in trajectories[0]:
                    agent.feed(ts)

                # Evaluate the performance. Play with random agents.
                if episode % args.evaluate_every == 0:
                    logger.log_performance(
                        episode,
                        tournament(
                            env,
                            args.num_eval_games,
                        )[0]
                    )

            # Get the paths
            csv_path = logger.csv_path
            fig_path = os.path.join(logger.log_dir, f'fig_rand_prob_{random_prob}.png')

        # Plot the learning curve
        plot_curve(csv_path, fig_path, args.algorithm)
        
        os.rename(csv_path, os.path.join(logger.log_dir, f'performance_rand_prob_{random_prob}.csv'))

        # Save model
        save_path = os.path.join(args.log_dir, f'model_random_p{random_prob}.pth')
        torch.save(agent, save_path)
        print(f'DQN Model against rule-based with random prob {random_prob} saved in', save_path)

        # remove the used rule-based model
        agents.pop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN RL for limited holdem")
    parser.add_argument(
        '--env',
        type=str,
        default='limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/limit_holdem_dqn_result/',
    )
    
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=-1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    rand_prob_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    
    train(args, random_prob_list=[1.0], curriculum=False)

