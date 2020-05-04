import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



class DuelingAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super(DuelingAgent, self).__init__(state_size, action_size, seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


class DoubleDQNAgent(Agent):
    """Interacts with and learns from the environment."""

    """
    def __init__(self, state_size, action_size, seed):
        super(DoubleDQNAgent, self).__init__(state_size, action_size, seed)
    """

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        next_state_actions = self.qnetwork_local(next_states).max(1)[1]
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_state_actions.unsqueeze(1))

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get predicted Q values from local model
        Q_predicted = self.qnetwork_local(states).gather(1, actions)

#        self.qnetwork_local.train()

        # Compute loss
        loss = F.mse_loss(Q_predicted, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()

        # Optimize
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


class TDErrorBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, epsilon=0.0001):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            epsilon (int): epsilon
        """
        self.action_size = action_size
        self.memory = []  # deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        self.epsiron = epsilon
        self.updated = False

    def add(self, td_error):
        """Add a TD error to memory."""
        self.memory.append(td_error)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def get_prioritized_indices(self, batch_size):
        '''Get indices prioritized by TD error.'''

        # Calculate the sum of TD errors
        # sum_absolute_td_error = np.sum(np.absolute(np.array(self.memory)))
        try:
            sum_absolute_td_error = np.sum(abs(np.array(self.memory)))
            sum_absolute_td_error += self.epsiron * len(self.memory)
        except Exception as e:
            print("self.memory.shape:", self.memory.shape)
            raise e

        # Creat random values of batch_size and sort them.
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # Try to get index using the sorted random values.
        indices = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                try:
                    tmp_sum_absolute_td_error += (
                            abs(self.memory[idx]) + self.epsiron)
                except Exception as e:
                    print("self.memory.shape:", self.memory.shape)
                    print("idx:", idx)
                    print("self.memory[idx]:", self.memory[idx])
                    print("self.memory:", self.memory)
                    raise e
                idx += 1

            # As I added epsion value, index can be over the index.
            # In case of that, I will decrement the index.
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indices.append(idx)

        return indices

    def update(self, td_errors):
        self.memory = td_errors
        self.updated = True

    def is_updated(self):
        return self.updated


class MoreAdvancedAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super(MoreAdvancedAgent, self).__init__(state_size, action_size, seed)
        # TD error memory
        self.td_error_memory = TDErrorBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def convert(self, experiences):
        """Convert a batch of experiences from memory."""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:

                # Until enough error momory is stored, use sample
                if self.td_error_memory.is_updated():
                    indices = self.td_error_memory.get_prioritized_indices(BATCH_SIZE)
                    experiences = [self.memory.memory[idx] for idx in indices]
                    states, actions, rewards, next_states, dones = self.convert(experiences)
                else:
                    experiences = self.memory.sample()
                    states, actions, rewards, next_states, dones = experiences

                self.learn((states, actions, rewards, next_states, dones), GAMMA)

    def update_td_error_memory(self):
        '''Update TD errors in the memory'''

        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

        # Make a batch using all memory
        experiences = self.memory.memory

        if len(experiences) is 0:
            return

        states, actions, rewards, next_states, dones = self.convert(experiences)

        # Obtain Q(s_t, a_t) from main/local network
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Carculate TD errors
        td_errors = (rewards + GAMMA * Q_targets_next * (1 - dones)) - \
                    Q_expected

        # Update TD errors - Tensor to NumPy to List
        self.td_error_memory.update(np.squeeze(td_errors.cpu().detach().numpy()).tolist())

    def done(self):
        """Update TD Error memory after each episode."""
        self.update_td_error_memory()