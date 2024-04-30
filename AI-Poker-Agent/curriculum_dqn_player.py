from pypokerengine.players import BasePokerPlayer
from time import sleep
from collections import OrderedDict
import numpy as np
import os
import torch

act_to_idx = {'call': 0,
              'raise': 1,
              'fold': 2,
              'check': 3
}
idx_to_act = {0:'call',
              1:'raise',
              2:'fold',
              3:'check'
}

def card_to_index(card):
    # Define the suits and their order
    suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    # Define the ranks and their order
    ranks = {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, 
             '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12}

    # Extract the suit and rank from the card
    if card[1].isdigit() and card[2:].isdigit():  # Check if it's a 10
        suit = card[0]
        rank = card[1:]
    else:
        suit = card[0]
        rank = card[1]

    # Calculate the index
    suit_index = suits[suit]
    rank_index = ranks[rank]

    # Each suit has 13 cards
    card_index = 13 * suit_index + rank_index
    return card_index

# build input state for DQN model
def build_state(valid_actions, hole_card, round_state):
    
    valid_ac = [a['action'] for a in valid_actions]
    legal_actions = OrderedDict({act_to_idx[a] : None for a in valid_ac})


    ex_state = {}
    obs = np.zeros(72)

    ex_state['legal_actions'] = legal_actions

    community_cards = round_state['community_card']
    hand_card = hole_card


    raise_nums = [0, 0, 0, 0]
    action_histories = round_state['action_histories']

    for round, actions in action_histories.items():
        raise_count = 0
        for ac in actions:
            if ac['action'] == 'RAISE':
                raise_count += 1
        if round == 'preflop':                  # preflop 
            raise_nums[0] = raise_count
        elif round == 'flop':                   # flop
            raise_nums[1] = raise_count
        elif round == 'turn':                   # turn
            raise_nums[2] = raise_count
        elif round == 'river':                  # river
            raise_nums[3] = raise_count
        else:
            raise ValueError
        
    
    cards = community_cards + hand_card

    idx = [card_to_index(card) for card in cards]
    obs[idx] = 1
    
    for i, num in enumerate(raise_nums):
        obs[52 + i * 5 + num] = 1
    
    ex_state['obs'] = obs
    ex_state['raw_legal_actions'] = [a for a in valid_ac]
    
    return ex_state


def load_model():
    model_path = 'limit_holdem_dqn_curriculum_v0/model_random_p0.0.pth'
    agent = None
    device = torch.device("cuda:0")
    if os.path.isfile(model_path):
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    
    return agent
        

class CurriculumDQNPlayer(BasePokerPlayer):

    def __init__(self):
        self.dqn_agent = load_model()

    def declare_action(self, valid_actions, hole_card, round_state):

        state = build_state(valid_actions, hole_card, round_state) 

        act_idx, _ = self.dqn_agent.eval_step(state)

        return idx_to_act[act_idx] # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def setup_ai():
        return CurriculumDQNPlayer()