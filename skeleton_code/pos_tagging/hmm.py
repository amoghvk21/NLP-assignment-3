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
    def __init__(self, num_states, num_obs, device=None):
        """
        For N hidden states and M observations,
            transition_prob: (N+1) * (N+1), with [0, :] as initial probabilities
            emission_prob: N * M

        Parameters:
            num_states: number of hidden states (POS tags)
            num_obs: number of observations
            device: Device to run model on
        """
        self.num_states = num_states
        self.num_obs = num_obs
        
        # Device handling: auto detect CUDA if available
        # otherwise use CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # MANUAL DEVICE SETTING TO CPU for testing
        self.device = torch.device('cpu')

        logger.info(f"Using device: {self.device}")
        
        self.epsilon = 1e-5
        self.transition_prob = torch.full(
            [self.num_states + 1, self.num_states + 1], self.epsilon, device=self.device
        )
        self.transition_prob[:, 0] = 0.0
        self.emission_prob = torch.full([self.num_states, self.num_obs], self.epsilon, device=self.device)
        self.log_scale = False
        
        # Variables for staged sEM training (persist across stages)
        self.sem_k = None  # Counter of batches performed - used for eta function
        self.sem_global_trans_stats = None  # Global transition statistics (mu)
        self.sem_global_emit_stats = None  # Global emission statistics (mu)

    def reset_logspace_random(self):
        """
        Reset the model parameters randomly from self.epsilon to 1 and normalise.

        """

        transition_init = torch.rand(self.num_states + 1, self.num_states + 1, device=self.device) + self.epsilon
        transition_init[:, 0] = 0.0         # cant transition to start state
        transition_init = transition_init / transition_init.sum(dim=1, keepdim=True)    # normalise
            
        emission_init = torch.rand(self.num_states, self.num_obs, device=self.device) + self.epsilon
        emission_init = emission_init / emission_init.sum(dim=1, keepdim=True)    # normalise

        self.transition_prob = torch.log(transition_init + self.epsilon)   # add epsilon just incase to avoid log(0) = -inf
        self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state
        
        self.emission_prob = torch.log(emission_init + self.epsilon)   # add epsilon just incase to avoid log(0) = -inf
        
        self.log_scale = True


    def reset_logspace_dirichlet(self):
        """
        Reset the model parameters using samples from the Dirichlet distribution with alpha=0.5. 

        Sample each row of matricies from this distribution.
        """

        alpha = 0.5

        
        dist_trans = torch.distributions.Dirichlet(
            torch.full((self.num_states + 1,), alpha, device=self.device)
        )
        trans_linear = dist_trans.sample((self.num_states + 1,))
        trans_linear = trans_linear + self.epsilon
        trans_linear = trans_linear / trans_linear.sum(dim=1, keepdim=True)
        self.transition_prob = torch.log(trans_linear)
        self.transition_prob[:, 0] = float("-inf")     # cant transition to the start state

        dist_emit = torch.distributions.Dirichlet(
            torch.full((self.num_obs,), alpha, device=self.device)
        )
        emit_linear = dist_emit.sample((self.num_states,))
        emit_linear = emit_linear + self.epsilon
        emit_linear = emit_linear / emit_linear.sum(dim=1, keepdim=True)
        self.emission_prob = torch.log(emit_linear)

        self.log_scale = True


    def reset_logspace(self, method: str = "dirichlet"):
        """
        Reset the model parameters using the specified method.
        """

        if method == "dirichlet":
            self.reset_logspace_dirichlet()
        elif method == "random":
            self.reset_logspace_random()
        else:
            raise ValueError("Invalid method name")


    def reset_uniform(self, log_space: bool):
        """
        Reset the model parameters to uniform distribution.
        Used by MLE
        """
        
        self.transition_prob = torch.full(
            [self.num_states + 1, self.num_states + 1], self.epsilon, device=self.device
        )
        self.transition_prob[:, 0] = 0.0

        self.emission_prob = torch.full([self.num_states, self.num_obs], self.epsilon, device=self.device)
        
        self.log_scale = log_space

        if log_space:
            self.transition_prob = torch.log(self.transition_prob)
            self.transition_prob[:, 0] = float("-inf")
            self.emission_prob = torch.log(self.emission_prob)


    def train(
        self,
        inputs: Dataset,
        epochs: int = 5,
        method: str = "mle",
        continue_training=False,
        initial_guesses=None,
        reset_method: str = "dirichlet",
        alpha_sem: float = 0.6,
    ) -> None:
        if method == "mle":
            self.train_logmle(inputs)
        elif method == "EM":
            self.train_EM_log(
                inputs,
                num_iter=epochs,
                continue_training=continue_training,
                initial_guesses=initial_guesses,
                reset_method = reset_method
            )
        elif method == "sEM":
            self.train_sEM(
                inputs,
                num_iter=epochs,
                eta_fn=lambda k: (k + 2) ** (-alpha_sem),
                continue_training=continue_training,
                initial_guesses=initial_guesses,
                reset_method = reset_method,
            )
        elif method == "hardEM":
            self.train_EM_hard_log(
                inputs,
                num_iter=epochs,
                continue_training=continue_training,
                initial_guesses=initial_guesses,
                reset_method = reset_method,
            )
        else:
            raise ValueError("Invalid training method name")

    def inference(self, input_ids) -> list:
        return self.viterbi_log(input_ids)


    @staticmethod
    def _normalize(mat):
        row_sums = mat.sum(dim=1, keepdim=True)

        mask_zero = (row_sums == 0)

        if mask_zero.any():
            logger.error(f"Fixed {mask_zero.sum().item()} rows with 0 sum (set to 1.0)")
        
        row_sums[mask_zero] = 1.0    # so that the denominator is 1 and not 0.    so the row of 0s stays 0s and isnt divided by 0
        
        return mat / row_sums


    @staticmethod
    def _log_normalize(log_matrix):
        """
        Expects a log matrix
        And normalises this
        """

        return log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)


    @staticmethod
    def _normalize_log(mat):
        """
        Expects a matrix (not log probs)
        And normalises this 
        And returns log probs
        """

        row_sums = mat.sum(dim=1, keepdim=True)  # (num_states + 1, 1)
        
        mask_zero = (row_sums == 0)    # (num_states + 1,)
        
        if mask_zero.any():
            logger.error(f"Fixed {mask_zero.sum().item()} rows with 0 sum (set to -inf)")

        row_sums[mask_zero] = 1.0  # as log(1) = 0    so we do log(row sum) - log(1)   =    log(0) - log(1)   =    -inf  - 0   =    -inf
        
        return torch.log(mat) - torch.log(row_sums)

    def train_mle(self, inputs: Dataset):
        """
        Supervised training by MLE
        """
        logger.info("Resetting model to uniform probs")
        self.reset_uniform(log_space=False)

        logger.info("Running MLE")

        assert not self.log_scale

        for sentence in tqdm(inputs, "MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]

            # Update initial probabilities
            # Changed this due to the 0th col being the start state probs
            # so the actual tags start from + 1
            self.transition_prob[0, tags[0] + 1] += 1

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
        
        logger.info("Resetting model to uniform probs")
        self.reset_uniform(log_space=False)
        
        logger.info("Running log-scale MLE")
        
        assert not self.log_scale

        for sentence in tqdm(inputs, "Log-MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]

            # Update initial probabilities
            # Changed this due to the 0th col being the start state probs
            # so the actual tags start from + 1
            self.transition_prob[0, tags[0] + 1] += 1

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
        reset_method: str = "dirichlet",
    ):
        """
        Train an HMM with the standard EM algorithm
        """

        # stage 1 or not using staged training
        if not continue_training:
            if initial_guesses is not None:
                logger.info("Soft EM: Using initial guesses for parameter initialization")
                self.transition_prob, self.emission_prob = initial_guesses
                # Ensure tensors are on the correct device
                self.transition_prob = self.transition_prob.to(self.device)
                self.emission_prob = self.emission_prob.to(self.device)
                # Assume initial_guesses are always in log space
                self.log_scale = True
            else:
                logger.info(f"Soft EM: Initializing parameters using {reset_method} method")
                self.reset_logspace(method=reset_method)
        
        if not self.log_scale:
            self.transition_prob = torch.log(self.transition_prob + self.epsilon)
            self.emission_prob = torch.log(self.emission_prob + self.epsilon)
            self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state
            self.log_scale = True

        for i in range(num_iter):
            logger.info(f"Soft EM iteration {i + 1}/{num_iter}")

            # Soft count accumulators (float)
            soft_trans_counts = torch.full(
                [self.num_states + 1, self.num_states + 1],
                fill_value=self.epsilon,
                dtype=torch.float32,
                device=self.device
            )
            soft_trans_counts[:, 0] = 0.0 # Impossible to transition to start state
            
            soft_emit_counts = torch.full(
                [self.num_states, self.num_obs],
                fill_value=self.epsilon,
                dtype=torch.float32,
                device=self.device
            )

            for sentence in tqdm(inputs, desc=f"Soft EM E-step"):
                input_ids = sentence["input_ids"]
                
                # Skip empty sentences
                if len(input_ids) == 0:
                    continue

                # Get expected counts for this sentence using forward backward
                trans_expected, emit_expected = self._forward_backward_counts(input_ids)
                
                # Accumulate into global counts
                soft_trans_counts += trans_expected
                soft_emit_counts += emit_expected

            # M-step - update parameters from soft counts
            self.transition_prob = self._normalize_log(soft_trans_counts)
            self.emission_prob = self._normalize_log(soft_emit_counts)


    def train_EM_hard_log(
        self,
        inputs: Dataset,
        num_iter: int = 10,
        initial_guesses=None,
        continue_training=False,
        reset_method: str = "dirichlet",
    ):
        """
        Train an HMM with the hard EM algorithm.
        """

        # stage 1 or not using staged training
        if not continue_training:
            if initial_guesses is not None:
                logger.info("Hard EM: Using initial guesses for parameter initialization")
                self.transition_prob, self.emission_prob = initial_guesses
                # Ensure tensors are on the correct device
                self.transition_prob = self.transition_prob.to(self.device)
                self.emission_prob = self.emission_prob.to(self.device)
                # Assume initial_guesses are always in log space
                self.log_scale = True
            else:
                logger.info(f"Hard EM: Initializing parameters using {reset_method} method")
                self.reset_logspace(method=reset_method)
        
        if not self.log_scale:
            self.transition_prob = torch.log(self.transition_prob + self.epsilon)
            self.emission_prob = torch.log(self.emission_prob + self.epsilon)
            self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state
            self.log_scale = True

        for iter in range(num_iter):
            logger.info(f"Hard EM iteration {iter+1}/{num_iter}")
            transition_counts = torch.full(
                [self.num_states + 1, self.num_states + 1], self.epsilon, device=self.device     # + 1 due to start state
            )

            # emissions are 0 indexed
            transition_counts[:, 0] = 0.0    # Impossible to transition to the start state
            emission_counts = torch.full(
                [self.num_states, self.num_obs], self.epsilon, device=self.device    # no start state as start state has no emissions
            )

            for sentence in tqdm(inputs, desc="hard EM with random initialisation"):
                input_ids = sentence["input_ids"]
                
                # Skip empty sentences
                if len(input_ids) == 0:
                    continue
                
                path = self.viterbi_log(input_ids)  # hidden state indices (0 indexed)

                # Conver to tensors
                input_ids_tensor = torch.tensor(input_ids, device=self.device, dtype=torch.long)
                path_tensor = torch.tensor(path, device=self.device, dtype=torch.long)

                # emission_counts[path[t], input_ids[t]] += 1    for all t in [0, len-1] at once
                emission_counts.index_put_(
                    (path_tensor, input_ids_tensor), 
                    torch.tensor(1.0, device=self.device), 
                    accumulate=True
                )
                
                # from_indices = [0, path[0]+1, path[1]+1, ... path[T-1]+1]   +1 due to start state at index 0 and path being zero indexed
                start_idx = torch.tensor([0], device=self.device, dtype=torch.long)
                from_indices = torch.cat([start_idx, path_tensor[:-1] + 1])    # dont need last state as doesnt transition to anything
                
                # to_indices = [path[0]+1, path[1]+1, ... path[T]+1]   +1 due to path being zero indexed
                to_indices = path_tensor + 1

                # transition_counts[from_indices, to_indices] += 1    for all t in [0, len-1] at once
                transition_counts.index_put_(
                    (from_indices, to_indices),
                    torch.tensor(1.0, device=self.device),
                    accumulate=True
                )


            # M-step - update parameters from hard counts
            self.transition_prob = self._normalize_log(transition_counts)
            self.emission_prob = self._normalize_log(emission_counts)


    def train_sEM(
        self,
        inputs: Dataset,
        num_iter: int = 30,
        eta_fn: Callable[[int], float]=None,
        initial_guesses=None,
        continue_training=False,
        batch_size: int = 30,
        reset_method: str = "dirichlet",
    ):
        """
        Train an HMM with a stepwise online EM algorithm
        """

        # stage 1 or not using staged training
        if not continue_training:
            if initial_guesses is not None:
                logger.info("Stepwise EM: Using initial guesses for parameter initialization")
                self.transition_prob, self.emission_prob = initial_guesses
                # Ensure tensors are on the correct device
                self.transition_prob = self.transition_prob.to(self.device)
                self.emission_prob = self.emission_prob.to(self.device)
                # Assume initial_guesses are always in log space
                self.log_scale = True
            else:
                logger.info(f"Stepwise EM: Initializing parameters using {reset_method} method")
                self.reset_logspace(method=reset_method)
        
        if not self.log_scale:
            self.transition_prob = torch.log(self.transition_prob + self.epsilon)
            self.emission_prob = torch.log(self.emission_prob + self.epsilon)
            self.transition_prob[:, 0] = float("-inf")  # Can't transition to start state
            self.log_scale = True

        # Only initialise when starting a new training (not continuing)
        if not continue_training or self.sem_global_trans_stats is None:
            # Initialise mu
            self.sem_global_trans_stats = torch.zeros(self.num_states + 1, self.num_states + 1, device=self.device)
            self.sem_global_emit_stats = torch.zeros(self.num_states, self.num_obs, device=self.device)
            self.sem_k = 0  # Counter of batches performed - used for eta function

        # Allocating memory here and zeroing at each batch for efficiency
        batch_trans_stats = torch.zeros(self.num_states + 1, self.num_states + 1, device=self.device)
        batch_emit_stats = torch.zeros(self.num_states, self.num_obs, device=self.device)

        logger.info("Fixed global stats persisting across epochs")
        logger.info("Stepwise EM training")
        logger.info(f"Number of epochs: {num_iter}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Eta function: {eta_fn}")
        logger.info(f"Initial guesses: {initial_guesses}")
        logger.info(f"Continue training: {continue_training}")
        logger.info(f"Reset method: {reset_method}")

        # Iterate over epochs
        for epoch in range(num_iter):
            logger.info(f"Stepwise EM epoch {epoch + 1}/{num_iter}")

            # Shuffle input dataset for each epoch
            indices = list(range(len(inputs)))
            random.shuffle(indices)

            # Create batches
            batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]

            # Iterate over batches
            for batch_idxs in tqdm(batches, desc=f"Epoch {epoch+1} batches"):

                # Stats for this batch
                batch_trans_stats.zero_()
                batch_emit_stats.zero_()

                valid_sentence_found = False

                # Iterate over sentences in current batch
                for idx in batch_idxs:
                    sentence = inputs[idx]
                    input_ids = sentence["input_ids"]
                    
                    # Skip empty sentences
                    if len(input_ids) == 0:
                        continue

                    # Run forward backward for sentence to get expected counts
                    trans_expected, emit_expected = self._forward_backward_counts(input_ids)   # sufficient stats

                    # Accumulate for each batch
                    batch_trans_stats += trans_expected
                    batch_emit_stats += emit_expected
                    valid_sentence_found = True

                # Skip updates if batch was empty
                if not valid_sentence_found:
                    continue

                # Compute stepsize
                stepsize = eta_fn(self.sem_k)

                # Interpolate global stats and local (sentence) stats
                self.sem_global_trans_stats = ((1 - stepsize) * self.sem_global_trans_stats) + (stepsize * batch_trans_stats)
                self.sem_global_emit_stats = ((1 - stepsize) * self.sem_global_emit_stats) + (stepsize * batch_emit_stats)

                self.sem_k += 1
                
                # Update parameters after each sentence
                # Add epsilon smoothing to avoid zero probabilities                
                # Normalise into valid log probabilities
                self.transition_prob = self._normalize_log(self.sem_global_trans_stats + self.epsilon)
                self.emission_prob = self._normalize_log(self.sem_global_emit_stats + self.epsilon)


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
            return torch.zeros(self.num_states, 0, device=self.device)
        
        num_states = self.num_states
        
        # log_alpha shape: (num_states, T)
        log_alpha = torch.zeros(num_states, T, device=self.device)
        
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
            return torch.zeros(self.num_states, 0, device=self.device)
        
        num_states = self.num_states

        # log_beta shape: (num_states, T)
        log_beta = torch.zeros(num_states, T, device=self.device)
        
        # Initialization: log_beta[:, T-1] = log(1) = 0

        # Precompute transitions
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


    def _forward_backward_counts(self, input_ids):
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
                torch.zeros(self.num_states + 1, self.num_states + 1, device=self.device),
                torch.zeros(self.num_states, self.num_obs, device=self.device)
            )
        
        # Run forward-backward
        log_alpha = self._forward_log(input_ids)   # (num_states, T)
        log_beta = self._backward_log(input_ids)   # (num_states, T)
        
        # Compute normalizer (log probability of the sequence)
        log_O = torch.logsumexp(log_alpha[:, T - 1], dim=0)
        
        # Initialize expected counts
        trans_expected = torch.zeros(self.num_states + 1, self.num_states + 1, device=self.device)
        emit_expected = torch.zeros(self.num_states, self.num_obs, device=self.device)


        
        # gamma 
        # Gamma storse how likely is the word at time t was generated by the pos tag s
        log_gamma = log_alpha + log_beta - log_O  # shape (num_states, T)

        gamma = torch.exp(log_gamma)  # shape (num_states, T)

        for t, observation in enumerate(input_ids):
            # gamma[:, t] is probability for each pos tag (all hidden states) at this step (for word at time t)
            emit_expected[:, observation] += gamma[:, t]


        # xi
        # Xi stores how likely to transition from pos tag (state) s to pos tag (state) s_prime at time t
        # P(y_t = s, y_{t+1} = s' | O, theta)
        # Initial transition (from start state to all states)
        # Initial transition (from start state to all states)
        log_xi0 = (
            self.transition_prob[0, 1 : self.num_states + 1]                   # (num_states,)
            + self.emission_prob[:, input_ids[0]]                              # (num_states,)
            + log_beta[:, 0]                                                   # (num_states,)
            - log_O                                                            # scalar
        )
        xi0 = torch.exp(log_xi0)  # (num_states,)
        trans_expected[0, 1:self.num_states + 1] += xi0  # add initial transitions to counts

        # Transitions between states
        # Prepare emission for all next observations: (T-1, num_states)
        observations = torch.tensor(input_ids[1:], dtype=torch.long, device=self.device)                   # (T-1,)
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
            - log_O                                     # scalar
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

        V = torch.zeros(N, seq_len, device=self.device)
        path = {}   # Dictionary to store the optimal path for each state at each time step

        # init 
        V[:, 0] = self.transition_prob[0, 1:] * self.emission_prob[:, input_ids[0]]  # Initial probabilities of going from start state to each other state

        # transition_matrix shape (N, N)
        transitions = self.transition_prob[1:, 1:] # from 1...N to 1...N

        for t in range(1, seq_len):  # Skip the first time step
            # prev_V shape (N, 1)
            prev_V = V[:, t-1].reshape(-1, 1)

            # emission shape (1, N) for token at t-1
            emission = self.emission_prob[:, input_ids[t]].reshape(1, -1)

            # For each current state y (index 0..N-1), calculate:
            #   V[y, t] = max_{y'} V[y', t-1] * trans[y', y] * emit[y, x_t]
            #   path[y, t] = argmax_{y'} (above)
            scores = prev_V * transitions * emission

            # Column-wise max and argmax operations
            best_scores, best_prev_indices = torch.max(scores, dim=0)
            V[:, t] = best_scores

            # path dict stores backpointers: for each state y at time t, store previous state (1..N, t)
            for y in range(N):  # y is 1..N
                path[(y, t)] = best_prev_indices[y].item()  # store as state index 1..N

        # Backtrace to find the optimal path
        last_state = torch.argmax(V[:, -1]).item()
        optimal_path = [last_state]

        current_state = last_state
        for t in range(seq_len - 1, 0, -1):
            current_state = path[current_state, t]
            optimal_path.insert(0, current_state) # prepend the state index

        return optimal_path


    def viterbi_log(self, input_ids):
        """
        Run Viterbi algorithm with log-scale probabilities
        """
        assert self.log_scale

        seq_len = len(input_ids)
        N = self.num_states

        # Handle empty sequences
        if seq_len == 0:
            return []

        V = torch.full((N, seq_len), float('-inf'), device=self.device)
        
        # Dictionary to store the optimal path for each state at each time step
        backpointers = torch.zeros((N, seq_len), dtype=torch.long, device=self.device)

        # init 
        V[:, 0] = self.transition_prob[0, 1:] + self.emission_prob[:, input_ids[0]]  # Initial probabilities of going from start state to each other state

        # transition_matrix shape (N, N)
        transitions = self.transition_prob[1:, 1:] # from 1...N to 1...N

        for t in range(1, seq_len):  # Skip the first time step
            # prev_V shape (N, 1)
            prev_V = V[:, t-1].reshape(-1, 1)

            # emission shape (1, N) for token at t-1
            emission = self.emission_prob[:, input_ids[t]].reshape(1, -1)

            # For each current state y (index 0..N-1), calculate:
            #   V[y, t] = max_{y'} V[y', t-1] * trans[y', y] * emit[y, x_t]
            #   path[y, t] = argmax_{y'} (above)
            scores = prev_V + transitions + emission

            # Column-wise max and argmax operations
            best_scores, best_prev_indices = torch.max(scores, dim=0)
            V[:, t] = best_scores

            # for each state y at time t, store previous state (1..N, t)
            backpointers[:, t] = best_prev_indices

        # Backtrace to find the optimal path
        last_state = torch.argmax(V[:, -1]).item()
        optimal_path = [last_state]

        current_state = last_state
        for t in range(seq_len - 1, 0, -1):
            current_state = backpointers[current_state, t].item()
            optimal_path.insert(0, current_state) # prepend the state index

        return optimal_path