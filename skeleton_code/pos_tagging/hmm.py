import logging
from typing import Callable
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
import random

from pos_tagging.base import BaseUnsupervisedClassifier


logger = logging.getLogger()

class HMMClassifier(BaseUnsupervisedClassifier):
    def __init__(self, num_states, num_obs):
        """
        For N hidden states and M observations,
            transition_prob: (N+1) * (N+1), with [0, :] as initial probabilities
            emission_prob: N * M

        Parameters:
            num_states: number of hidden states
            num_obs: number of observations
        """
        self.num_states = num_states
        self.num_obs = num_obs
        # Initialized to epsilon, so allowing unseen transition/emission to have p>0
        self.epsilon = 1e-5
        self.transition_prob = torch.full(
            [self.num_states + 1, self.num_states + 1], self.epsilon
        )
        self.transition_prob[:, 0] = 0.0
        self.emission_prob = torch.full([self.num_states, self.num_obs], self.epsilon)
        self.log_scale = False
        self.cnt = 0  # Number of updates in sEM
        # TODO: optimize training by using UNK token

    def reset(self):
        self.transition_prob = torch.full(
            [self.num_states + 1, self.num_states + 1], self.epsilon
        )
        self.transition_prob[:, 0] = 0.0
        self.emission_prob = torch.full([self.num_states, self.num_obs], self.epsilon)
        self.log_scale = False
        self.cnt = 0

    def train(
        self,
        inputs: Dataset,
        epochs: int = 5,
        method: str = "mle",
        continue_training=False,
    ) -> None:
        if method == "mle":
            self.train_logmle(inputs)
        elif method == "EM":
            self.train_EM_log(
                inputs, num_iter=epochs, continue_training=continue_training
            )
        elif method == "sEM":
            self.train_sEM(
                inputs,
                num_iter=epochs,
                eta_fn=lambda k: (k + 2) ** (-1.0),
                continue_training=continue_training,
            )
        elif method == "hardEM":
            self.train_EM_hard_log(
                inputs, num_iter=epochs, continue_training=continue_training
            )
        else:
            raise ValueError("Invalid training method name")

    def inference(self, input_ids) -> list:
        return self.viterbi_log(input_ids)

    @staticmethod
    def _normalize(mat):
        for i in range(mat.size(0)):
            if torch.sum(mat[i]) == 0:
                continue
            mat[i] = mat[i] / torch.sum(mat[i])
        return mat

    @staticmethod
    def _log_normalize(log_matrix):
        return log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)

    @staticmethod
    def _normalize_log(mat):
        for i in range(mat.size(0)):
            if torch.sum(mat[i]) == 0:
                continue
            mat[i] = torch.log(mat[i]) - torch.log(torch.sum(mat[i]))
        return mat

    def train_mle(self, inputs: Dataset):
        """
        Supervised training by MLE
        """
        logger.info("Running MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]

            # Update initial probabilities
            self.transition_prob[0, tags[0]] += 1

            for i in range(len(input_ids)):
                # Update transition probabilities
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1

                # Update emission probabilities
                self.emission_prob[tags[i], input_ids[i]] += 1

        self.transition_prob = self._normalize(self.transition_prob)
        self.emission_prob = self._normalize(self.emission_prob)

    def train_logmle(self, inputs: Dataset):
        """Train with MLE algorithm using log likelihood to avoid underflow"""
        logger.info("Running log-scale MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "Log-MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]

            # Update initial probabilities
            self.transition_prob[0, tags[0]] += 1

            for i in range(len(input_ids)):
                # Update transition probabilities
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1

                # Update emission probabilities
                self.emission_prob[tags[i], input_ids[i]] += 1

        self.transition_prob = self._normalize_log(self.transition_prob)
        self.emission_prob = self._normalize_log(self.emission_prob)
        self.log_scale = True

    def train_EM_log(
        self,
        inputs: Dataset,
        num_iter: int = 5,
        initial_guesses=None,
        continue_training=False,
    ):
        """
        Train an HMM with the standard EM algorithm
        """

        # If not continuing training, reset to clean state
        if not continue_training:
            self.reset()
        # If continuing training but probabilities are in regular space then convert to log space
        elif not self.log_scale:
            self.transition_prob = torch.log(self.transition_prob + self.epsilon)
            self.emission_prob = torch.log(self.emission_prob + self.epsilon)
            self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state
        
        self.log_scale = True

        # Complete your code here

        # If initial_guesses is provided, use as initialisation; otherwise, randomize
        if initial_guesses is not None and not continue_training:
            self.transition_prob, self.emission_prob = initial_guesses
        elif not continue_training:
            # Uniform initialisation in log-space, with epsilon for smoothing
            self.transition_prob = torch.log(
                torch.full([self.num_states + 1, self.num_states + 1], fill_value=self.epsilon)
            )
            self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state
            self.emission_prob = torch.log(
                torch.full([self.num_states, self.num_obs], fill_value=self.epsilon)
            )

        for i in range(num_iter):
            logger.info(f"Soft EM iteration {i + 1}/{num_iter}")

            # Soft count accumulators (float!)
            soft_trans_counts = torch.full(
                [self.num_states + 1, self.num_states + 1],
                fill_value=0.0, dtype=torch.float32
            )
            soft_emit_counts = torch.full(
                [self.num_states, self.num_obs],
                fill_value=0.0, dtype=torch.float32
            )

            for sentence in tqdm(inputs, desc=f"Soft EM E-step"):
                input_ids = sentence["input_ids"]
                
                # Skip empty sentences
                if len(input_ids) == 0:
                    continue

                # Get expected counts for this sentence using forward backward
                trans_expected, emit_expected = self._forward_backward_sentence_counts(input_ids)
                
                # Accumulate into global counts
                soft_trans_counts += trans_expected
                soft_emit_counts += emit_expected

            # No zero probabilities
            soft_trans_counts = soft_trans_counts + self.epsilon
            soft_emit_counts = soft_emit_counts + self.epsilon
            
            # 5. M-step: update parameters from soft counts (use log version for log_scale)
            self.transition_prob = self._normalize_log(soft_trans_counts)
            self.emission_prob = self._normalize_log(soft_emit_counts)


    def train_EM_hard_log(
        self,
        inputs: Dataset,
        num_iter: int = 10,
        initial_guesses=None,
        continue_training=False,
    ):
        """
        Train an HMM with the hard EM algorithm (also called Viterbi EM)
        """

        # If not continuing training, reset to clean state
        if not continue_training:
            self.reset()
        # If continuing training but probabilities are in regular space then convert to log space
        elif not self.log_scale:
            self.transition_prob = torch.log(self.transition_prob + self.epsilon)
            self.emission_prob = torch.log(self.emission_prob + self.epsilon)
            self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state
        
        self.log_scale = True

        # Complete your code here

        # If initial_guesses is provided then use it
        if initial_guesses is not None and not continue_training:
            self.transition_prob, self.emission_prob = initial_guesses
        # otherwise randomise
        elif not continue_training:
            # Uniform initialisation in log-space, with epsilon for smoothing
            self.transition_prob = torch.log(
                torch.full([self.num_states + 1, self.num_states + 1], fill_value=self.epsilon)
            )
            self.transition_prob[:, 0] = float("-inf")  # Cant transition to start state
            self.emission_prob = torch.log(
                torch.full([self.num_states, self.num_obs], fill_value=self.epsilon)
            )

        for iter in range(num_iter):
            logger.info(f"Hard EM iteration {iter+1}/{num_iter}")
            transition_counts = torch.full(
                [self.num_states + 1, self.num_states + 1], self.epsilon     # + 1 due to start state
            )

            # emissions are 0 indexed
            transition_counts[:, 0] = 0.0    # Impossible to transition to the start state
            emission_counts = torch.full(
                [self.num_states, self.num_obs], self.epsilon    # no start state as start state has no emissions
            )

            for sentence in tqdm(inputs, desc="hard EM"):
                input_ids = sentence["input_ids"]
                
                # Skip empty sentences
                if len(input_ids) == 0:
                    continue
                
                path = self.viterbi_log(input_ids)  # hidden state indices (1-indexed: 1..N)

                # Handle initial transition and emission for first token as special case due to start state and no emissions
                # path[0] is 1-indexed (1..N) so convert to 0-indexed for emission_counts (0..N-1)
                # emissions are 0 indexed (0..N-1) as the dimensions are (N, M)
                transition_counts[0, path[0]] += 1     # initial transition; 0 due to start state
                emission_counts[path[0]-1, input_ids[0]] += 1    # emission for first token (convert 1-indexed to 0-indexed)

                for t in range(1, len(input_ids)):
                    # path and transitions are 1-indexed
                    transition_counts[path[t-1], path[t]] += 1
                    # emissions are 0 indexed (0..N-1) so minus 1
                    emission_counts[path[t]-1, input_ids[t]] += 1

            # M-step: update parameters from hard counts
            self.transition_prob = self._normalize_log(transition_counts)
            self.emission_prob = self._normalize_log(emission_counts)


    def train_sEM(
        self,
        inputs: Dataset,
        num_iter: int = 30,
        eta_fn: Callable[[int], float] = lambda k: 0.8,
        initial_guesses=None,
        continue_training=False,
    ):
        """
        Train an HMM with a stepwise online EM algorithm
        """

        # If not continuing training, reset to clean state
        if not continue_training:
            self.reset()
        # If continuing training but probabilities are in regular space then convert to log space
        elif not self.log_scale:
            self.transition_prob = torch.log(self.transition_prob + self.epsilon)
            self.emission_prob = torch.log(self.emission_prob + self.epsilon)
            self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state

        self.log_scale = True

        # If initial_guesses is provided then use it
        if initial_guesses is not None and not continue_training:
            self.transition_prob, self.emission_prob = initial_guesses
        # otherwise randomise
        elif not continue_training:
            # Uniform initialisation in log-space, with epsilon for smoothing
            self.transition_prob = torch.log(
                torch.full([self.num_states + 1, self.num_states + 1], fill_value=self.epsilon)
            )
            self.transition_prob[:, 0] = float("-inf")  # Cant transition to start state
            self.emission_prob = torch.log(
                torch.full([self.num_states, self.num_obs], fill_value=self.epsilon)
            )

        # Initialise
        global_trans_stats = torch.zeros(self.num_states + 1, self.num_states + 1)
        global_emit_stats = torch.zeros(self.num_states, self.num_obs)

        k = 0  # Counter of updates performed

        for epoch in range(num_iter):
            logger.info(f"Stepwise EM epoch {epoch + 1}/{num_iter}")

            # Shuffle input dataset for each epoch for stochasticity
            indices = list(range(len(inputs)))
            random.shuffle(indices)

            for idx in indices:
                sentence = inputs[idx]
                input_ids = sentence["input_ids"]
                
                # Skip empty sentences
                if len(input_ids) == 0:
                    continue

                # Run forward backward for sentence to get expected counts
                trans_expected, emit_expected = self._forward_backward_sentence_counts(input_ids)

                # Compute stepsize
                stepsize = eta_fn(k)

                # Interpolate global stats and local (sentence) stats
                global_trans_stats = ((1 - stepsize) * global_trans_stats) + (stepsize * trans_expected)
                global_emit_stats = ((1 - stepsize) * global_emit_stats) + (stepsize * emit_expected)

                k += 1
                
                # Update parameters after each sentence
                # Add epsilon smoothing to avoid zero probabilities                
                # Normalise into valid log probabilities
                self.transition_prob = self._normalize_log(global_trans_stats + self.epsilon)
                self.emission_prob = self._normalize_log(global_emit_stats + self.epsilon)


    def _forward_log(self, input_ids):
        """
        Compute forward probabilities in log space.
        log_alpha[s, t] = log P(x_1...x_t, y_t = s | theta)

        Args:
            input_ids: List of observation indices

        Returns:
            log_alpha: Tensor of shape (num_states, T) where
                log_alpha[s, t] = log P(x_1...x_t, y_t = s | theta)
        """
        T = len(input_ids)
        if T == 0:
            # Return empty tensor for empty sequence
            return torch.zeros(self.num_states, 0)
        
        num_states = self.num_states
        
        # log_alpha shape: (num_states, T)
        log_alpha = torch.zeros(num_states, T)
        
        # Initialisation for t=0
        # log_alpha[:, 0] = log P(y_0 = s, x_0 | theta)
        # prob of start state transitioning to each state and emitting the first observation
        log_alpha[:, 0] = (
            self.transition_prob[0, 1:num_states + 1]  # (num_states,)
            + self.emission_prob[:, input_ids[0]]      # (num_states,)
        )

        # transitions shape: (num_states, num_states)
        # got rid of start state as not needed
        transitions = self.transition_prob[1:num_states + 1, 1:num_states + 1]

        # Main dp 
        for t in range(1, T):
            # log_alpha: shape (num_states, prev_states)
            #     log_alpha[:, t-1][j] + transitions[j, s]
            # We want logsumexp over j (for each s)
            prev_alpha = log_alpha[:, t - 1].reshape(-1, 1)            # (num_states, 1)
            emission = self.emission_prob[:, input_ids[t]]             # (num_states,)

            # log_probs: (num_states, num_states)
            log_probs = prev_alpha + transitions

            log_alpha[:, t] = torch.logsumexp(log_probs, dim=0) + emission  # sum over all prev states to get 1 val per next state

        return log_alpha


    def _backward_log(self, input_ids):
        """
        Compute backward probabilities in log space.
        
        Args:
            input_ids: List of observation indices
            
        Returns:
            log_beta: Tensor of shape (num_states, T) where
                log_beta[s, t] = log P(x_{t+1}...x_T | y_t = s, theta)
        """
        T = len(input_ids)
        if T == 0:
            # Return empty tensor for empty sequence
            return torch.zeros(self.num_states, 0)
        
        num_states = self.num_states

        # log_beta shape: (num_states, T)
        log_beta = torch.zeros(num_states, T)
        
        # Initialization: log_beta[:, T-1] = log(1) = 0

        # Precompute transitions for speed
        # transitions: (num_states, num_states)
        transitions = self.transition_prob[1:num_states + 1, 1:num_states + 1]  # [from_state, to_state]
        
        # main dp loop
        for t in range(T - 2, -1, -1):
            # log_beta (num_states, next_states)
            prev_beta = log_beta[:, t + 1].reshape(1, -1)   # (1, num_states)
            emission = self.emission_prob[:, input_ids[t + 1]].reshape(1, -1)   # (1, num_states)
                    
            # log_probs (num_states, num_states)
            log_probs = transitions + prev_beta + emission   # shape: (num_states, num_states)

            log_beta[:, t] = torch.logsumexp(log_probs, dim=1)  # sum over all next states to get 1 val per prev state
        
        return log_beta


    def _forward_backward_sentence_counts(self, input_ids):
        """
        Compute expected transition and emission counts for a single sentence using forward backward.
        
        Args:
            input_ids: List of observation indices
            
        Returns:
            trans_expected: Tensor of shape (num_states + 1, num_states + 1) with expected transition counts
            emit_expected: Tensor of shape (num_states, num_obs) with expected emission counts
        """
        T = len(input_ids)
        
        # Handle empty sequences
        if T == 0:
            return (
                torch.zeros(self.num_states + 1, self.num_states + 1),
                torch.zeros(self.num_states, self.num_obs)
            )
        
        # Run forward-backward
        log_alpha = self._forward_log(input_ids)   # (num_states, T)
        log_beta = self._backward_log(input_ids)   # (num_states, T)
        
        # Compute normalizer (log probability of the sequence)
        logZ = torch.logsumexp(log_alpha[:, T - 1], dim=0)
        
        # Initialize expected counts
        trans_expected = torch.zeros(self.num_states + 1, self.num_states + 1)
        emit_expected = torch.zeros(self.num_states, self.num_obs)
        
        # Accumulate soft emission counts (gamma)
        # Gamma storse how likely is the word at time t was generated by the pos tag s
        log_gamma = log_alpha + log_beta - logZ  # shape (num_states, T)

        gamma = torch.exp(log_gamma)  # shape (num_states, T)

        for t, observation in enumerate(input_ids):
            # gamma[:, t] is probability for each pos tag (all hidden states) at this step (for word at time t)
            emit_expected[:, observation] += gamma[:, t]

        # Accumulate soft transition counts (xi)
        # Xi stores how likely to transition from pos tag (state) s to pos tag (state) s_prime at time t
        # Initial transition (from start state to all states)
        # Initial transition (from start state to all states)
        log_xi0 = (
            self.transition_prob[0, 1 : self.num_states + 1]                   # (num_states,)
            + self.emission_prob[:, input_ids[0]]                              # (num_states,)
            + log_beta[:, 0]                                                   # (num_states,)
            - logZ                                                             # scalar
        )
        xi0 = torch.exp(log_xi0)  # (num_states,)
        trans_expected[0, 1:self.num_states + 1] += xi0  # add initial transitions to counts

        # Transitions between states (vectorised over all time steps)
        # Prepare emission for all next observations: (T-1, num_states)
        observations = torch.tensor(input_ids[1:])                   # (T-1,)
        emission = self.emission_prob[:, observations].T             # (T-1, num_states)

        # Reshaping all
        log_alpha_t = log_alpha[:, :T-1].T.reshape(T-1, self.num_states, 1)             # (T-1, num_states, 1)
        transition = self.transition_prob[1:self.num_states+1, 1:self.num_states+1]
        transition = transition.reshape(1, self.num_states, self.num_states)            # (1, num_states, num_states)
        emission = emission.reshape(T-1, 1, self.num_states)                            # (T-1, 1, num_states)
        log_beta_tplus1 = log_beta[:, 1:T].T.reshape(T-1, 1, self.num_states)           # (T-1, 1, num_states)

        # log_xi shape (T-1, num_states, num_states)
        log_xi = (
            log_alpha_t                                 # (T-1, num_states, 1)
            + transition                                # (1, num_states, num_states)
            + emission                                  # (T-1, 1, num_states)
            + log_beta_tplus1                           # (T-1, 1, num_states)
            - logZ                                      # scalar
        )

        xi = torch.exp(log_xi)  # (T-1, num_states, num_states)

        # Sum over all time steps to get expected counts for each transition (from state s to state s_prime for all t)
        trans_expected[1:, 1:] += xi.sum(dim=0)  # (num_states, num_states)

        return trans_expected, emit_expected

    def viterbi(self, input_ids):
        """Run Viterbi algorithm"""
        assert not self.log_scale

        seq_len = len(input_ids)
        N = self.num_states

        # Handle empty sequences
        if seq_len == 0:
            return []

        V = torch.zeros(N + 1, seq_len + 1)
        path = {}   # Dictionary to store the optimal path for each state at each time step
        V[:, 0] = self.transition_prob[0, :]  # Initial probabilities of going from start state to each other state

        # Complete code here

        for t in range(1, seq_len + 1):  # Skip the first time step
            # prev_V shape (N,)
            prev_V = V[1:, t-1]

            # transition_matrix shape (N, N)
            transition = self.transition_prob[1:, 1:] # from 1...N to 1...N

            # emission shape (N,) for token at t-1
            emission = self.emission_prob[:, input_ids[t-1]]

            # For each current state y (index 0..N-1), calculate:
            #   V[y, t] = max_{y'} V[y', t-1] * trans[y', y] * emit[y, x_t]
            #   path[y, t] = argmax_{y'} (above)
            scores = prev_V.unsqueeze(1) * transition * emission.unsqueeze(0)  # (N, N)

            # Column-wise max and argmax operations
            V[1:, t], argmax_x = torch.max(scores, dim=0)

            # path dict stores backpointers: for each state y at time t, store previous state (1..N, t)
            for y in range(1, N+1):  # y is 1..N
                path[(y, t)] = argmax_x[y-1] + 1  # store as state index 1..N

        # Backtrace to find the optimal path
        optimal_path = []
        last_state = (torch.argmax(V[1:, len(input_ids)]) + 1).item()
        optimal_path.append(last_state - 1)  # manually add the first state index

        for t in range(len(input_ids), 1, -1):
            last_state = path[last_state.item(), t]
            optimal_path.insert(0, last_state.item()) # prepend the state index

        return optimal_path


    def viterbi_log(self, input_ids):
        """
        Run Viterbi algorithm with log-scale probabilities
        """
        assert self.log_scale

        seq_len = len(input_ids)
        N = self.num_states  # number of hidden states

        # Handle empty sequences
        if seq_len == 0:
            return []

        # V is (N+1, seq_len+1)
        V = torch.full((N + 1, seq_len + 1), float("-inf"))
        path = {}
        V[:, 0] = self.transition_prob[0, :]  # Initiial probs of going from start state to each other state

        # Complete code here

        for t in range(1, seq_len + 1):  # Skip the first time step
            # prev_V shape (N,)
            prev_V = V[1:, t-1]
            
            # transition_matrix shape (N, N)
            transition = self.transition_prob[1:, 1:] # from 1...N to 1...N

            # emission shape (N,) for token at t-1
            emission = self.emission_prob[:, input_ids[t-1]]

            # For each current state y (index 0..N-1), calculate:
            #     max_x { prev_V[x] + transition[x, y] + emission[y] }
            # (N, 1) + (N, N) + (1, N) -> (N, N)
            scores = prev_V.reshape(-1, 1) + transition + emission.reshape(1, -1)

            # Column-wise max and argmax operations
            V[1:, t], argmax_x = torch.max(scores, dim=0)

            # path dict stores backpointers: for each state y at time t, store previous state (1..N, t)
            for y in range(1, N+1):  # y is 1..N
                path[(y, t)] = argmax_x[y-1] + 1  # store as state index 1..N

        # Backtrace to find the optimal path
        optimal_path = []
        last_state = torch.argmax(V[1:, -1]) + 1 # V stores index 1...N for real states
        optimal_path.append(last_state.item()) # manually add the first state index

        for t in range(len(input_ids), 1, -1):
            last_state = path[last_state.item(), t]
            optimal_path.insert(0, last_state.item()) # prepend the state index

        return optimal_path