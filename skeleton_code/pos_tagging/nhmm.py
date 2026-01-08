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
        nn.init.normal_(self.state_embeddings.weight, mean=0.0, std=1.0)

        # init word embeddings to N(0,1)
        self.word_embeddings = self.emission_net[2]    # Linear layer (vocab_size, hidden_dim)
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=1.0)
        
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
        state_tensor = torch.tensor(state).to(self.device)
        logits = self.emission_net(state_tensor)    # (vocab_size,)
        log_probs = F.log_softmax(logits, dim=0) # turn into log probs
        return log_probs[word_idx]
    
    def _get_emission_log_matrix(self) -> torch.Tensor:
        """
        Compute emission log matrix.
        
        Returns:
            log_E_matrix: Tensor of shape (num_states, vocab_size) with emission log probabilities
        """
        
        all_state_indicies = torch.arange(self.num_states).to(self.device)
        logits = self.emission_net(all_state_indicies)   # Pass it in all states at once
        log_E_matrix = F.log_softmax(logits, dim=1)  # normalise across columns
        return log_E_matrix

    def _forward_log(self, input_ids: List[int]) -> torch.Tensor:
        """
        Compute forward probabilities in log space

        Args:
            input_ids: List of word indices

        Returns:
            log_alpha: Tensor of shape (num_states, T) where log_alpha[s, t] = log P(x_1...x_t, y_t = s | theta)
        """
        T = len(input_ids)
        if T == 0:
            return torch.zeros(self.num_states, 0).to(self.device)

        # Get all matricies
        log_T_matrix = self._get_transition_log_matrix()       # (num_states, num_states): log_T_matrix[i, j] = log P(j | i)
        log_E_matrix = self._get_emission_log_matrix()         # (num_states, vocab_size): log_E_matrix[s, v] = log P(v | s)
        initial_log_probs = self._get_initial_log_probs()      # (num_states,)

        log_alpha = torch.zeros(self.num_states, T, device=self.device)
        
        # Init - prob of starting in state and seeing first word 
        log_alpha[:, 0] = initial_log_probs + log_E_matrix[:, input_ids[0]]

        # Forward recursion
        for t in range(1, T):
            log_alpha_prev = log_alpha[:, t-1].reshape(-1, 1)  # (num_states, 1)
            scores = log_alpha_prev + log_T_matrix  # (num_states, 1) + (num_states, num_states) = (num_states, num_states)   -  broadcasting
            log_alpha[:, t] = torch.logsumexp(scores, dim=0) + log_E_matrix[:, input_ids[t]]   # (num_states,)
        
        return log_alpha
    
    def _backward_log(self, input_ids: List[int]) -> torch.Tensor:
        """
        Compute backward probabilities in log space using neural networks.
        
        Args:
            input_ids: List of word indices
            
        Returns:
            log_beta: Tensor of shape (num_states, T) where
                log_beta[s, t] = log P(x_{t+1}...x_T | y_t = s, theta)
        """
        T = len(input_ids)
        if T == 0:
            return torch.zeros(self.num_states, 0).to(self.device)

        log_beta = torch.zeros(self.num_states, T, device=self.device)
        # Initialization: log_beta[:, T-1] = log(1) = 0 (already initialized)

        # Get all matricies
        log_T_matrix = self._get_transition_log_matrix()  # (num_states, num_states): log_T_matrix[i, j] = log P(j | i)
        log_E_matrix = self._get_emission_log_matrix()    # (num_states, vocab_size): log_E_matrix[s, v] = log P(v | s)

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
    
    def _forward_backward(self, input_ids: List[int]) -> tuple:
        """
        Compute forward-backward probabilities and posteriors.
        
        Args:
            input_ids: List of word indices
            
        Returns:
            log_alpha: Forward probabilities (num_states, T)
            log_beta: Backward probabilities (num_states, T)
            log_gamma: State posteriors (num_states, T)
            log_xi: Transition posteriors (T-1, num_states, num_states)
            logZ: Log probability of sequence
        """
        T = len(input_ids)
        num_states = self.num_states
        device = self.device
        if T == 0:
            return (
                torch.zeros(num_states, 0).to(device),
                torch.zeros(num_states, 0).to(device),
                torch.zeros(num_states, 0).to(device),
                torch.zeros(0, num_states, num_states).to(device),
                torch.tensor(0.0).to(device)
            )
        
        log_alpha = self._forward_log(input_ids)  # (num_states, T)
        log_beta = self._backward_log(input_ids)  # (num_states, T)
        
        # Compute normalizer (log probability of the sequence) - just sum the last column to get all states
        logZ = torch.logsumexp(log_alpha[:, T - 1], dim=0)

        # Compute state posteriors: gamma[s, t] = P(y_t = s | x)
        log_gamma = log_alpha + log_beta - logZ  # (num_states, T)

        
        # Compute transition posteriors: xi[t, s, s'] = log P(y_t = s, y_{t+1} = s' | x)
        # prob at some time t, prob of from state s to state s'

        # Get transition and emission log matrices
        log_T_matrix = self._get_transition_log_matrix()  # (num_states, num_states)
        log_E_matrix = self._get_emission_log_matrix()    # (num_states, vocab_size)

        # Prepare emission for all next observations: (T-1, num_states)
        next_obs = torch.tensor(input_ids[1:], device=device)  # (T-1,)
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
        
        # Use Adam optimizer with specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # only sentences of 40 words or less
        filtered_dataset = []
        for example in dataset:
            forms = example["form"]
            if len(forms) > 0 and len(forms) <= max_sentence_length:
                filtered_dataset.append(example)
        logger.info(f"Filtered dataset: {len(filtered_dataset)} sentences (max length {max_sentence_length})")
        
        for epoch in range(max_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Process in minibatches
            for batch_start in tqdm(range(0, len(filtered_dataset), minibatch_size), desc=f"Epoch {epoch+1}/{max_epochs}"):
                batch_end = min(batch_start + minibatch_size, len(filtered_dataset))
                batch = filtered_dataset[batch_start:batch_end]
                
                prev_log_prob = None
                for inner_iter in range(max_inner_loops):
                    batch_loss = 0.0
                    total_log_prob = 0.0
                    
                    optimizer.zero_grad()
                    
                    # Process each sentence in batch
                    for example in batch:
                        forms = example["form"]
                        if len(forms) == 0:
                            continue
                        
                        # Convert words to indices (with digit normalization)
                        input_ids = [self._get_word_idx(word) for word in forms]
                        
                        # Forward-backward to get posteriors
                        log_alpha, log_beta, log_gamma, log_xi, logZ = self._forward_backward(input_ids)
                        
                        # Compute expected log-likelihood (negative log-likelihood as loss)
                        # Loss = -log P(x) = -logZ   - we are maximising the prob of seeing the observed sequence
                        loss = -logZ
                        batch_loss += loss
                        total_log_prob += logZ.item()
                    
                    # Average loss over batch
                    avg_batch_loss = batch_loss / len(batch)
                    
                    # Backpropagate
                    avg_batch_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    
                    # stop if log prob change < convergence_threshold
                    if prev_log_prob is not None:
                        log_prob_change = abs(total_log_prob - prev_log_prob)    # difference between the 2 losses
                        if log_prob_change < convergence_threshold:
                            logger.debug(f"Converged at inner iter {inner_iter+1}: log prob change = {log_prob_change:.6f}")
                            break
                    
                    prev_log_prob = total_log_prob
                
                total_loss += avg_batch_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch+1}/{max_epochs}: Average loss = {avg_loss:.4f}")
    
    def viterbi_log(self, input_ids: List[int]) -> List[int]:
        """
        Run Viterbi algorithm to find most likely state sequence.
        
        Args:
            input_ids: List of word indices
            
        Returns:
            path: List of state indices (0-indexed) for most likely path
        """
        T = len(input_ids)
        S = self.num_states

        if T == 0:
            return []

        device = self.device

        # V stores log probabilities: V[s, t] = max log P(x_1...x_t, y_1...y_t, y_t = s)
        V = torch.full((S, T), float('-inf'), device=device)
        backpointers = torch.zeros((S, T), dtype=torch.long, device=device)

        # Precompute all emissions for the sequence
        log_E_matrix = self._get_emission_log_matrix().to(device)  # (num_states, vocab_size)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        emission_log_probs = log_E_matrix[:, input_ids_tensor].T  # (T, num_states)

        # Precompute all state transition log probs as a matrix
        transition_log_probs = self._get_transition_log_matrix().to(device)  # (S, S)

        # Initialization
        initial_log_probs = self._get_initial_log_probs().to(device)  # (S,)

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
        word_ids = [self._get_word_idx(word) for word in words]

        return self.viterbi_log(word_ids)
    
    def evaluate(self, dataset: Dataset) -> dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        all_true_tags = []
        all_pred_tags = []
        
        for example in tqdm(dataset, desc="Evaluating"):
            words = example["form"]
            true_tags = example["tags"]
            
            if len(words) == 0:
                continue
            
            # Predict tags
            pred_tags = self.inference(words)
            
            if len(true_tags) != len(pred_tags):
                raise ValueError(
                    f"Length mismatch detected!\n"
                    f"Input Words: {len(words)}\n"
                    f"True Tags:   {len(true_tags)}\n"
                    f"Pred Tags:   {len(pred_tags)}\n"
                    f"Sentence:    {words}\n"
                    "Check your Viterbi implementation or Data Loader."
                )
            
            all_true_tags.extend(true_tags)
            all_pred_tags.extend(pred_tags)
        
        return {
            "true_tags": all_true_tags,
            "pred_tags": all_pred_tags
        }
