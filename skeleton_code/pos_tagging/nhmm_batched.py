import logging
from typing import Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
import random

from pos_tagging.base import BaseUnsupervisedClassifier


logger = logging.getLogger()


class NeuralHMMClassifier(nn.Module, BaseUnsupervisedClassifier):   
    def __init__(self, num_states: int, vocab: List, tag_mapping: dict, device: torch.device):
        """
        Initialise Neural HMM classifier
        
        Args:
            num_states: Number of hidden states (POS tags)
            vocab: List of sentences (list of dicts with "form" key) for building vocabulary
            tag_mapping: Dictionary mapping tag strings to integers
            device: Device to run model on (cuda/cpu)   
        """
        nn.Module.__init__(self)
        BaseUnsupervisedClassifier.__init__(self)

        self.num_states = num_states
        self.device = device
        
        self.vocab_set = set()
        for sentence in vocab:
            for word in sentence["form"]:
                self.vocab_set.add(self._normalise_word(word))
                
        self.vocab_list = sorted(list(self.vocab_set))
        self.vocab_size = len(self.vocab_list)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab_list)}
        
        # Used for word embeddings, state embeddings, and hidden layers
        self.hidden_dim = 512
        
        # Transition network
        # logits = U . q  +  b
        # softmax over rows
        self.query_vector = nn.Parameter(torch.randn(self.hidden_dim).to(device))    # (hidden_dim,)
        self.transition_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_states * self.num_states)
        ).to(device)
        self._init_linear_layers(self.transition_net)
        
        # Initial state network: outputs log probabilities for initial states
        self.initial_param = nn.Parameter(torch.randn(num_states).to(device))
        nn.init.normal_(self.initial_param, mean=0.0, std=1.0)
        
        # Emission network
        self.emission_net = nn.Sequential(
            nn.Embedding(num_states, self.hidden_dim),   # Lookup Tag Vector
            nn.ReLU(),                                   # Non-linearity
            nn.Linear(self.hidden_dim, self.vocab_size)  # Projection to Vocab
        ).to(device)
        self._init_linear_layers(self.emission_net)

        # State embeddings (one embedding per hidden state)
        # init state embeddings to N(0,1)
        self.state_embeddings = self.emission_net[0]    # Embedding layer (num_states, hidden_dim)
        # nn.init.normal_(self.state_embeddings.weight, mean=0.0, std=1.0)       # if commented out then initialised to uniform distribution using init linear layers

        # init word embeddings to N(0,1)
        self.word_embeddings = self.emission_net[2]    # Linear layer (vocab_size, hidden_dim)
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=1.0)       # if commented out then initialised to uniform distribution using init linear layers
        
        self.epsilon = 1e-8
        
        logger.info(f"Initialised NeuralHMM with {self.num_states} states, {self.vocab_size} vocabulary size, hidden_dim={self.hidden_dim}")
    
    def _init_linear_layers(self, sequential_net):
        """
        Initialise linear layers with Uniform distribution. 
        Mean = 0, std = sqrt(1/n_in)
        """
        for module in sequential_net:
            if isinstance(module, nn.Linear):
                n_in = module.weight.size(1)
                std = np.sqrt(1.0 / n_in)
                nn.init.uniform_(module.weight, a=-std * np.sqrt(3), b=std * np.sqrt(3))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @staticmethod
    def _normalise_word(word: str) -> str:
        """
        Map all digits to 0
        Used for preprocessing
        """
        normalised = ''.join('0' if c.isdigit() else c for c in word)
        return normalised
    
    def _get_word_idx(self, word: str) -> int:
        """
        Get word index.
        Return 0 for unknown words.
        """
        normalised_word = self._normalise_word(word)
        return self.word_to_idx.get(normalised_word, 0)
    
    def _get_initial_log_probs(self) -> torch.Tensor:
        """
        Compute initial state log probabilities using learnable parameter.
        
        Returns:
            log_probs: Tensor of shape (num_states,) with log probabilities
        """
        # Use learnable parameter vector for initial probabilities
        log_probs = F.log_softmax(self.initial_param, dim=0)
        return log_probs
    
    def _get_transition_log_probs(self, prev_state: int) -> torch.Tensor:
        """
        Compute transition log probabilities from previous state.
        
        Args:
            prev_state: Previous state index (0-indexed)
            
        Returns:
            log_probs: Tensor of shape (num_states,) with log probabilities for next states
        """
        
        flat_T = self.transition_net(self.query_vector)
        T_matrix = flat_T.view(self.num_states, self.num_states)  # Reshape into (num_states, num_states)
        log_T_matrix = F.log_softmax(T_matrix, dim=1)
        return log_T_matrix[prev_state]
    
    def _get_transition_log_matrix(self) -> torch.Tensor:
        """
        Compute transition log matrix.
        
        Returns:
            log_T_matrix: Tensor of shape (num_states, num_states) with transition log probabilities
        """

        flat_T = self.transition_net(self.query_vector)
        T_matrix = flat_T.view(self.num_states, self.num_states)  # Reshape into (num_states, num_states)
        log_T_matrix = F.log_softmax(T_matrix, dim=1)
        return log_T_matrix

    def _get_emission_log_prob(self, word_idx: int, state: int) -> torch.Tensor:
        """
        Compute emission log probability for word given state.
        
        Args:
            word_idx: Word index
            state: State index (0-indexed)
            
        Returns:
            log_prob: Scalar tensor with log probability
        """
        state_tensor = torch.tensor(state, device=self.device)
        logits = self.emission_net(state_tensor)    # (vocab_size,)
        log_probs = F.log_softmax(logits, dim=0) # turn into log probs
        return log_probs[word_idx]
    
    def _get_emission_log_matrix(self) -> torch.Tensor:
        """
        Compute emission log matrix.
        
        Returns:
            log_E_matrix: Tensor of shape (num_states, vocab_size) with emission log probabilities
        """
        
        all_state_indicies = torch.arange(self.num_states, device=self.device)
        logits = self.emission_net(all_state_indicies)   # Pass it in all states at once
        log_E_matrix = F.log_softmax(logits, dim=1)  # normalise across columns
        return log_E_matrix

    def _forward_log(
        self, 
        input_ids: torch.Tensor,
        log_T_matrix: torch.Tensor,
        log_E_matrix: torch.Tensor,
        initial_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward probabilities in log space

        Args:
            input_ids: List of word indices
            log_T_matrix: Precomputed transition log matrix (num_states, num_states)
            log_E_matrix: Precomputed emission log matrix (num_states, vocab_size)
            initial_log_probs: Precomputed initial log probabilities (num_states,)

        Returns:
            log_alpha: Tensor of shape (num_states, T) where log_alpha[s, t] = log P(x_1...x_t, y_t = s | theta)
        """
        T = input_ids.size(0)
        if T == 0:
            return torch.zeros(self.num_states, 0, device=self.device)

        log_alpha = torch.zeros(self.num_states, T, device=self.device)
        
        # Init - prob of starting in state and seeing first word 
        log_alpha[:, 0] = initial_log_probs + log_E_matrix[:, input_ids[0]]

        # Forward recursion
        for t in range(1, T):
            log_alpha_prev = log_alpha[:, t-1].reshape(-1, 1)  # (num_states, 1)
            scores = log_alpha_prev + log_T_matrix  # (num_states, 1) + (num_states, num_states) = (num_states, num_states)   -  broadcasting
            log_alpha[:, t] = torch.logsumexp(scores, dim=0) + log_E_matrix[:, input_ids[t]]   # (num_states,)
        
        return log_alpha
    
    def _backward_log(
        self,
        input_ids: torch.Tensor,
        log_T_matrix: torch.Tensor,
        log_E_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute backward probabilities in log space using neural networks.
        
        Args:
            input_ids: List of word indices
            log_T_matrix: Precomputed transition log matrix (num_states, num_states)
            log_E_matrix: Precomputed emission log matrix (num_states, vocab_size)
            
        Returns:
            log_beta: Tensor of shape (num_states, T) where
                log_beta[s, t] = log P(x_{t+1}...x_T | y_t = s, theta)
        """
        T = input_ids.size(0)
        if T == 0:
            return torch.zeros(self.num_states, 0, device=self.device)

        log_beta = torch.zeros(self.num_states, T, device=self.device)
        # Initialization: log_beta[:, T-1] = log(1) = 0 (already initialized)

        # Backward recursion
        for t in range(T - 2, -1, -1):

            emission = log_E_matrix[:, input_ids[t + 1]]   # (num_states,)
            emission = emission.view(1, -1)   # (1, num_states)

            next_beta = log_beta[:, t + 1].view(1, -1)   # (1, num_states)

            sum_matrix = (       # (num_states, num_states)
                log_T_matrix + 
                emission + 
                next_beta
            )

            log_beta[:, t] = torch.logsumexp(sum_matrix, dim=1) # sum over columns (next states) to get dim (num_states,)

        return log_beta
    
    def _forward_backward(
        self,
        input_ids: torch.Tensor,
        log_T_matrix: torch.Tensor,
        log_E_matrix: torch.Tensor,
        initial_log_probs: torch.Tensor
    ) -> tuple:
        """
        Compute forward-backward probabilities and posteriors.
        
        Args:
            input_ids: List of word indices
            log_T_matrix: Precomputed transition log matrix (num_states, num_states)
            log_E_matrix: Precomputed emission log matrix (num_states, vocab_size)
            initial_log_probs: Precomputed initial log probabilities (num_states,)
            
        Returns:
            log_alpha: Forward probabilities (num_states, T)
            log_beta: Backward probabilities (num_states, T)
            log_gamma: State posteriors (num_states, T)
            log_xi: Transition posteriors (T-1, num_states, num_states)
            logZ: Log probability of sequence
        """
        T = input_ids.size(0)
        num_states = self.num_states
        device = self.device
        if T == 0:
            return (
                torch.zeros(num_states, 0, device=device),
                torch.zeros(num_states, 0, device=device),
                torch.zeros(num_states, 0, device=device),
                torch.zeros(0, num_states, num_states, device=device),
                torch.tensor(0.0, device=device)
            )
        
        log_alpha = self._forward_log(input_ids, log_T_matrix, log_E_matrix, initial_log_probs)  # (num_states, T)
        log_beta = self._backward_log(input_ids, log_T_matrix, log_E_matrix)  # (num_states, T)
        
        # Compute normalizer (log probability of the sequence) - just sum the last column to get all states
        logZ = torch.logsumexp(log_alpha[:, T - 1], dim=0)

        # Compute state posteriors: gamma[s, t] = P(y_t = s | x)
        log_gamma = log_alpha + log_beta - logZ  # (num_states, T)

        
        # Compute transition posteriors: xi[t, s, s'] = log P(y_t = s, y_{t+1} = s' | x)
        # prob at some time t, prob of from state s to state s'

        # Prepare emission for all next observations: (T-1, num_states)
        next_obs = input_ids[1:]  # (T-1,)
        emission = log_E_matrix[:, next_obs].T  # (T-1, num_states)

        log_alpha_t = log_alpha[:, :T-1].T.reshape(T-1, num_states, 1)      # (T-1, num_states, 1)
        log_beta_tplus1 = log_beta[:, 1:T].T.reshape(T-1, 1, num_states)    # (T-1, 1, num_states)
        transition = log_T_matrix.reshape(1, num_states, num_states)        # (1, num_states, num_states)
        emission = emission.reshape(T-1, 1, num_states)                     # (T-1, 1, num_states)

        # log_xi shape (T-1, num_states, num_states)
        log_xi = (
            log_alpha_t          # (T-1, num_states, 1)
            + transition         # (1, num_states, num_states)
            + emission           # (T-1, 1, num_states)
            + log_beta_tplus1    # (T-1, 1, num_states)
            - logZ
        )
        # (T-1, num_states, num_states) reuslt

        return log_alpha, log_beta, log_gamma, log_xi, logZ
    
    def _forward_log_batched(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        log_T_matrix: torch.Tensor,
        log_E_matrix: torch.Tensor,
        initial_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward probabilities in log space for a batch of sequences.

        Args:
            input_ids: Padded word indices (batch_size, max_seq_len)
            lengths: Actual lengths of each sequence (batch_size,)
            log_T_matrix: Precomputed transition log matrix (num_states, num_states)
            log_E_matrix: Precomputed emission log matrix (num_states, vocab_size)
            initial_log_probs: Precomputed initial log probabilities (num_states,)

        Returns:
            log_alpha: Tensor of shape (batch_size, num_states, max_seq_len)
        """
        batch_size, max_T = input_ids.shape
        num_states = self.num_states
        device = self.device

        # Initialize with -inf (log(0))
        log_alpha = torch.full((batch_size, num_states, max_T), float('-inf'), device=device)
        
        # Get emissions for all positions: (batch_size, max_T, num_states)
        # log_E_matrix is (num_states, vocab_size), input_ids is (batch_size, max_T)
        emissions = log_E_matrix[:, input_ids].permute(1, 2, 0)  # (batch_size, max_T, num_states)
        
        # Init: log_alpha[:, :, 0] = initial_log_probs + emissions[:, 0, :]
        log_alpha[:, :, 0] = initial_log_probs.unsqueeze(0) + emissions[:, 0, :]  # (batch_size, num_states)

        # Forward recursion
        for t in range(1, max_T):
            # log_alpha[:, :, t-1] is (batch_size, num_states)
            log_alpha_prev = log_alpha[:, :, t-1].unsqueeze(2)  # (batch_size, num_states, 1)
            # log_T_matrix is (num_states, num_states) - broadcast over batch
            scores = log_alpha_prev + log_T_matrix.unsqueeze(0)  # (batch_size, num_states, num_states)
            # Sum over previous states (dim=1)
            log_alpha[:, :, t] = torch.logsumexp(scores, dim=1) + emissions[:, t, :]  # (batch_size, num_states)

        return log_alpha
    
    def _backward_log_batched(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        log_T_matrix: torch.Tensor,
        log_E_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute backward probabilities in log space for a batch of sequences.

        Args:
            input_ids: Padded word indices (batch_size, max_seq_len)
            lengths: Actual lengths of each sequence (batch_size,)
            log_T_matrix: Precomputed transition log matrix (num_states, num_states)
            log_E_matrix: Precomputed emission log matrix (num_states, vocab_size)

        Returns:
            log_beta: Tensor of shape (batch_size, num_states, max_seq_len)
        """
        batch_size, max_T = input_ids.shape
        num_states = self.num_states
        device = self.device

        # Initialize with -inf, then set valid end positions to 0
        log_beta = torch.full((batch_size, num_states, max_T), float('-inf'), device=device)
        
        # Set beta at actual sequence end to 0 for each sequence (vectorized)
        # For sequences of length L, beta[i, :, L-1] = 0
        end_positions = lengths - 1  # (batch_size,)
        # Create indices: for each (batch_i, state_j), set position end_positions[batch_i]
        bi = torch.arange(batch_size, device=device).repeat_interleave(num_states)  # [0,0,..,0, 1,1,..,1, ...]
        si = torch.arange(num_states, device=device).repeat(batch_size)              # [0,1,..,S-1, 0,1,..,S-1, ...]
        ti = end_positions.repeat_interleave(num_states)                             # [end0,end0,.., end1,end1,..]
        log_beta[bi, si, ti] = 0.0
        
        # Get emissions for all positions: (batch_size, max_T, num_states)
        emissions = log_E_matrix[:, input_ids].permute(1, 2, 0)  # (batch_size, max_T, num_states)

        # Backward recursion - process all timesteps, masking will handle invalid ones
        for t in range(max_T - 2, -1, -1):
            # Only update positions where t < length - 1
            # emission at t+1: (batch_size, num_states)
            emission = emissions[:, t + 1, :]  # (batch_size, num_states)
            emission = emission.unsqueeze(1)   # (batch_size, 1, num_states)
            
            # next_beta: (batch_size, num_states) -> (batch_size, 1, num_states)
            next_beta = log_beta[:, :, t + 1].unsqueeze(1)  # (batch_size, 1, num_states)
            
            # log_T_matrix: (num_states, num_states) -> (1, num_states, num_states)
            transition = log_T_matrix.unsqueeze(0)  # (1, num_states, num_states)
            
            # sum_matrix: (batch_size, num_states, num_states)
            sum_matrix = transition + emission + next_beta
            
            # Sum over next states (dim=2)
            new_beta = torch.logsumexp(sum_matrix, dim=2)  # (batch_size, num_states)
            
            # Only update where t < length - 1 (i.e., t+1 is valid)
            mask = (t < lengths - 1).unsqueeze(1)  # (batch_size, 1)
            log_beta[:, :, t] = torch.where(mask, new_beta, log_beta[:, :, t])

        return log_beta
    
    def _forward_backward_batched(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        log_T_matrix: torch.Tensor,
        log_E_matrix: torch.Tensor,
        initial_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward-backward for a batch and return logZ (log probability of sequences).

        Args:
            input_ids: Padded word indices (batch_size, max_seq_len)
            lengths: Actual lengths of each sequence (batch_size,)
            log_T_matrix: Precomputed transition log matrix (num_states, num_states)
            log_E_matrix: Precomputed emission log matrix (num_states, vocab_size)
            initial_log_probs: Precomputed initial log probabilities (num_states,)

        Returns:
            logZ: Log probability of each sequence (batch_size,)
        """
        batch_size = input_ids.shape[0]
        device = self.device

        # Forward pass
        log_alpha = self._forward_log_batched(
            input_ids, lengths, log_T_matrix, log_E_matrix, initial_log_probs
        )  # (batch_size, num_states, max_seq_len)

        # Compute logZ for each sequence using its actual length (vectorized)
        # logZ[i] = logsumexp(log_alpha[i, :, lengths[i]-1])
        batch_indices = torch.arange(batch_size, device=device)
        end_positions = lengths - 1  # (batch_size,)
        # log_alpha[:, :, end_positions] doesn't work directly, need advanced indexing
        log_alpha_ends = log_alpha[batch_indices, :, end_positions]  # (batch_size, num_states)
        logZ = torch.logsumexp(log_alpha_ends, dim=1)  # (batch_size,)

        return logZ

    def train_model(
        self, 
        dataset: Dataset,
        max_epochs: int = 5,
        lr: float = 0.001, 
        minibatch_size: int = 256,
        max_inner_loops: int = 6,
        convergence_threshold: float = 1e-4,
        max_grad_norm: float = 5.0,
        max_sentence_length: int = 40,
    ):
        """
        Train the Neural HMM using Generalized EM (forward-backward + backpropagation).
        
        The gradient is computed as:
        J(θ) = Σ_z p(z | x) ∂/∂θ ln p(x, z | θ)
        
        Hyperparameters:
        - Epochs: 5
        - Minibatch size: 256 sentences
        - Inner loop updates: max 6 per minibatch
        - Convergence threshold: log prob change < 1e-4
        - Gradient clipping: norm > 5
        - Max sentence length: 40 words
        
        Args:
            dataset: Training dataset
            max_epochs: Maximum number of training epochs (default: 5)
            lr: Learning rate for optimizer
            minibatch_size: Number of sentences per batch (default: 256)
            max_inner_loops: Maximum inner loop updates per batch (default: 6)
            convergence_threshold: Stop if log prob change < this (default: 1e-4)
            max_grad_norm: Clip gradients if norm exceeds this (default: 5.0)
            max_sentence_length: Filter sentences longer than this (default: 40)
        """
        logger.info(f"Training Neural HMM for {max_epochs} epochs")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Minibatch size: {minibatch_size}, Max inner loops: {max_inner_loops}")
        logger.info(f"Gradient clipping: {max_grad_norm}, Max sentence length: {max_sentence_length}")
        logger.info(f"Device: {self.device}")
        
        # Use Adam optimizer with specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # only sentences of 40 words or less
        filtered_dataset = []
        for example in dataset:
            forms = example["form"]
            if len(forms) > 0 and len(forms) <= max_sentence_length:
                filtered_dataset.append(example)
        logger.info(f"Filtered dataset: {len(filtered_dataset)} sentences (max length {max_sentence_length})")
        
        # for epoch in range(max_epochs):
        for epoch in tqdm(range(max_epochs), desc="Total Training", leave=False):
            total_loss = torch.tensor(0.0, device=self.device)
            num_batches = 0
            
            # Process in minibatches
            for batch_start in tqdm(
                range(0, len(filtered_dataset), minibatch_size),
                desc=f"  Epoch {epoch+1}/{max_epochs}",
                leave=False
            ):
                batch_end = min(batch_start + minibatch_size, len(filtered_dataset))
                batch = filtered_dataset[batch_start:batch_end]
                
                # Prepare batched input: convert all sentences to padded tensor
                batch_input_ids = []
                batch_lengths = []
                for example in batch:
                    forms = example["form"]
                    if len(forms) == 0:
                        continue
                    word_ids = [self._get_word_idx(word) for word in forms]
                    batch_input_ids.append(word_ids)
                    batch_lengths.append(len(word_ids))
                
                if len(batch_input_ids) == 0:
                    continue
                
                # Pad sequences to max length in batch
                max_len = max(batch_lengths)
                padded_input_ids = torch.zeros(len(batch_input_ids), max_len, dtype=torch.long, device=self.device)
                for i, ids in enumerate(batch_input_ids):
                    padded_input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                lengths = torch.tensor(batch_lengths, dtype=torch.long, device=self.device)
                
                actual_batch_size = len(batch_input_ids)
                
                prev_log_prob = None
                for inner_iter in range(max_inner_loops):
                    optimizer.zero_grad()
                    
                    # PRECOMPUTE matrices once for all sentences in this batch
                    log_T_matrix = self._get_transition_log_matrix()  # (num_states, num_states)
                    log_E_matrix = self._get_emission_log_matrix()    # (num_states, vocab_size)
                    initial_log_probs = self._get_initial_log_probs()  # (num_states,)
                    
                    # Batched forward-backward to get logZ for all sentences at once
                    logZ_batch = self._forward_backward_batched(
                        padded_input_ids, lengths, log_T_matrix, log_E_matrix, initial_log_probs
                    )  # (batch_size,)
                    
                    # Compute loss: negative log probability
                    total_log_prob = logZ_batch.sum()
                    batch_loss = -total_log_prob
                    
                    # Average loss over batch
                    avg_batch_loss = batch_loss / actual_batch_size
                    
                    # Backpropagate
                    avg_batch_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    
                    # stop if log prob change < convergence_threshold
                    if prev_log_prob is not None:
                        log_prob_change_tensor = torch.abs(total_log_prob - prev_log_prob)
                        if log_prob_change_tensor.item() < convergence_threshold:
                            logger.debug(f"Converged at inner iter {inner_iter+1}")
                            break
                    
                    prev_log_prob = total_log_prob.detach()
                
                total_loss += avg_batch_loss
                num_batches += 1
            
            # avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            # logger.info(f"Epoch {epoch+1}/{max_epochs}: Average loss = {avg_loss:.4f}")
    
    def viterbi_log(self, input_ids: torch.Tensor) -> List[int]:
        """
        Run Viterbi algorithm to find most likely state sequence.
        
        Args:
            input_ids: Tensor of word indices
            
        Returns:
            path: List of state indices (0-indexed) for most likely path
        """
        T = input_ids.size(0)
        S = self.num_states

        if T == 0:
            return []

        device = self.device

        # V stores log probabilities: V[s, t] = max log P(x_1...x_t, y_1...y_t, y_t = s)
        V = torch.full((S, T), float('-inf'), device=device)
        backpointers = torch.zeros((S, T), dtype=torch.long, device=device)

        # Precompute all emissions for the sequence
        log_E_matrix = self._get_emission_log_matrix()  # (num_states, vocab_size)
        emission_log_probs = log_E_matrix[:, input_ids].T  # (T, num_states)

        # Precompute all state transition log probs as a matrix
        transition_log_probs = self._get_transition_log_matrix()  # (S, S)

        # Initialization
        initial_log_probs = self._get_initial_log_probs()  # (S,)

        V[:, 0] = initial_log_probs + emission_log_probs[0]

        # Viterbi recursion
        for t in range(1, T):
            # previous V: (S,)
            prev_V = V[:, t - 1].reshape(-1, 1)  # (S,1)
            scores = prev_V + transition_log_probs  # (S,S)
            
            # take max over previous states for each current state
            best_scores, best_prev = torch.max(scores, dim=0)  # (S,)   and   (S,)

            V[:, t] = best_scores + emission_log_probs[t]
            backpointers[:, t] = best_prev

        # Backtrace
        path = []
        
        # Find best final state
        best_final = torch.argmax(V[:, T-1]).item()
        path.append(best_final)

        # Trace back
        current_state = best_final
        for t in range(T - 1, 0, -1):
            current_state = backpointers[current_state, t].item()
            path.insert(0, current_state)  # insert at front

        return path
    
    def inference(self, words) -> List[int]:
        """
        Run Viterbi algorithm to find most likely state sequence.
        
        Args:
            words: List of word strings
            
        Returns:
            List of predicted state ids (0-indexed)
        """
        # Convert words to indices
        word_ids = torch.tensor(
            [self._get_word_idx(word) for word in words],
            dtype=torch.long,
            device=self.device
        )

        return self.viterbi_log(word_ids)
    
    # def evaluate(self, dataset: Dataset) -> dict:
    #     """
    #     Evaluate the model on a dataset.
        
    #     Args:
    #         dataset: Evaluation dataset
            
    #     Returns:
    #         Dictionary with evaluation results
    #     """
    #     results = []
    #     all_true_tags = []
    #     all_pred_tags = []
        
    #     for example in tqdm(dataset, desc="Evaluating"):
    #         words = example["form"]
    #         true_tags = example["tags"]
            
    #         if len(words) == 0:
    #             continue
            
    #         # Predict tags
    #         pred_tags = self.inference(words)
            
    #         if len(true_tags) != len(pred_tags):
    #             raise ValueError(
    #                 f"Length mismatch detected!\n"
    #                 f"Input Words: {len(words)}\n"
    #                 f"True Tags:   {len(true_tags)}\n"
    #                 f"Pred Tags:   {len(pred_tags)}\n"
    #                 f"Sentence:    {words}\n"
    #                 "Check your Viterbi implementation or Data Loader."
    #             )
            
    #         all_true_tags.extend(true_tags)
    #         all_pred_tags.extend(pred_tags)
        
    #     return {
    #         "true_tags": all_true_tags,
    #         "pred_tags": all_pred_tags
    #     }
