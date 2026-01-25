import logging
from typing import Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
import random
import csv

from pos_tagging.base import BaseUnsupervisedClassifier
from utils import calculate_variation_of_information, calculate_v_measure


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

        # Initialise all linear layers using uniform distribution
        std = np.sqrt(1.0 / self.hidden_dim)
        nn.init.uniform_(self.transition_net[0].weight, a=-std * np.sqrt(3), b=std * np.sqrt(3))
        nn.init.zeros_(self.transition_net[0].bias)

        # Initialise query vector to N(0,1) as its described as a embedding in the paper
        nn.init.normal_(self.query_vector, mean=0.0, std=1.0)
        
        # Initial state network: outputs log probabilities for initial states
        self.initial_param = nn.Parameter(torch.randn(num_states).to(device))
        nn.init.normal_(self.initial_param, mean=0.0, std=1.0)
        
        # Emission network
        self.emission_net = nn.Sequential(
            nn.Embedding(num_states, self.hidden_dim),   # Lookup Tag Vector
            nn.ReLU(),                                   # Non-linearity
            nn.Linear(self.hidden_dim, self.vocab_size)  # Projection to Vocab
        ).to(device)

        # init state embeddings to N(0,1) as embedding layers are initialised using gaussian
        self.state_embeddings = self.emission_net[0]    # Embedding layer (num_states, hidden_dim)
        nn.init.normal_(self.state_embeddings.weight, mean=0.0, std=1.0)

        # init word embeddings to N(0,1) as per paper
        self.word_embeddings = self.emission_net[2]    # Linear layer (vocab_size, hidden_dim)
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=1.0)

        # init word biases to 0
        nn.init.zeros_(self.word_embeddings.bias)
        
        self.epsilon = 1e-8
        
        logger.info(f"Initialised NeuralHMM with {self.num_states} states, {self.vocab_size} vocabulary size, hidden_dim={self.hidden_dim}")
    
    
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

    def compute_metrics(self, dataset: Dataset) -> dict:
        """
        Compute clustering metrics (normalized VI, homogeneity, completeness, V-score) 
        for the dataset.
        
        Args:
            dataset: Dataset with "form" (words) and "upos" (true labels)
            
        Returns:
            Dictionary with metrics: normalized_vi, homogeneity, completeness, v_score
        """
        all_true_tags = []
        all_pred_tags = []
        num_samples = len(dataset)
        
        self.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc="Computing metrics", total=num_samples, leave=False):
                forms = example["form"]
                true_tags = example["tags"]
                
                if len(forms) == 0:
                    continue
                
                # Get predictions
                pred_tags = self.inference(forms)
                
                all_true_tags.extend(true_tags)
                all_pred_tags.extend(pred_tags)
        self.train()
        
        # Compute metrics
        homogeneity, completeness, v_score = calculate_v_measure(all_true_tags, all_pred_tags)
        _, normalized_vi = calculate_variation_of_information(all_true_tags, all_pred_tags)
        
        return {
            "normalized_vi": normalized_vi,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_score": v_score
        }

    def _forward_log_batched(
        self,
        input_ids_padded: torch.Tensor,
        lengths: torch.Tensor,
        log_T_matrix: torch.Tensor,
        log_E_matrix: torch.Tensor,
        initial_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward probabilities in log space for a batch of sequences.
        log_alpha = log P(x_1...x_t, y_t = s | theta)
        prob of seeing sequence x_1 ... x_t and being in state s at time t given the model parameters

        Args:
            input_ids_padded: Padded word indices (batch_size, max_len)
            lengths: Actual lengths of each sequence (batch_size,)
            log_T_matrix: Precomputed transition log matrix (num_states, num_states)
            log_E_matrix: Precomputed emission log matrix (num_states, vocab_size)
            initial_log_probs: Precomputed initial log probabilities (num_states,)

        Returns:
            log_O: Tensor of shape (batch_size,) with log probability of each sequence
        """
        batch_size, max_len = input_ids_padded.shape
        S = self.num_states

        if max_len == 0:
            return torch.zeros(batch_size, device=self.device)

        # log_alpha = log P(x_1...x_t, y_t = s | theta)   -  prob of seeing sequence 1 ... t and being in state s at time t given the model parameters
        # log_alpha: (batch_size, num_states, max_len)
        log_alpha = torch.full((batch_size, S, max_len), float('-inf'), device=self.device)

        # log_E_matrix: (num_states, vocab_size)
        # input_ids_padded: (batch_size, max_len)
        emissions = log_E_matrix[:, input_ids_padded]  # (num_states, batch_size, max_len)
        emissions = emissions.permute(1, 2, 0)  # (batch_size, max_len, num_states)

        # Init - prob of starting in state s and seeing first word
        # initial_log_probs: (num_states,)
        # emissions[:, 0, :]: (batch_size, num_states)
        log_alpha[:, :, 0] = initial_log_probs.reshape(1, -1) + emissions[:, 0, :]  # (batch_size, num_states)

        for t in range(1, max_len):
            # log_alpha_prev: (batch_size, num_states, 1)
            log_alpha_prev = log_alpha[:, :, t-1]     # (batch_size, num_states)
            log_alpha_prev = log_alpha_prev.reshape(batch_size, S, 1)   # (batch_size, num_states, 1)
            
            # scores: (batch_size, num_states, num_states)
            # log_T_matrix[i, j] = log P(state j | state i)
            # log_alpha_prev[b, i, 1] + log_T_matrix[i, j] = scores[b, i, j]
            scores = log_alpha_prev + log_T_matrix.reshape(1, S, S)     # (batch_size, num_states, 1) + (1, num_states, num_states) = (batch_size, num_states, num_states)
            
            # logsumexp over previous states (dim=1) to get prob of being in state s at time t given all previous states
            log_alpha[:, :, t] = torch.logsumexp(scores, dim=1) + emissions[:, t, :]  # (batch_size, num_states)


        # Extract log_O for each sequence at its actual length
        # lengths: (batch_size,)
        # log_alpha[b, :, lengths[b]-1] contains probs of all obs
        
        # length_indices: (batch_size, num_states, 1)
        # length - 1 to make indicies
        # expand to (batch_size, num_states, 1) to match log_alpha shape
        length_indices = (lengths - 1).reshape(batch_size, 1, 1).expand(batch_size, S, 1)

        # get probs of last indicies in all sequences using length indicies
        final_log_alpha = log_alpha.gather(2, length_indices).reshape(batch_size, S)  # (batch_size, num_states)
        
        
        # log_O: (batch_size,)
        # last col of alpha contains prob of being in state s at time t given all obs
        # need to sum over all states to get log prob of sequence (all obs)
        log_O = torch.logsumexp(final_log_alpha, dim=1)

        return log_O

    
    def _pad_batch(self, batch: List[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a batch of sentences to padded tensor and lengths.
        Allows for batching
        
        Args:
            batch: List of dicts containing word lists (observations)
            
        Returns:
            input_ids_padded: Padded word indices (batch_size, max_len)
            lengths: Actual lengths of each sequence (batch_size,)
        """
        # Convert all sentences to input_ids
        batch_input_ids = []  # (batch_size, sentence_len)
        for example in batch:
            forms = example["form"]
            input_ids = [self._get_word_idx(word) for word in forms]
            batch_input_ids.append(input_ids)
        
        # Get lengths and max_len
        lengths = torch.tensor([len(ids) for ids in batch_input_ids], dtype=torch.long, device=self.device)
        max_len = lengths.max().item()
        
        # Pad sequences (pad with 0)
        batch_size = len(batch_input_ids)
        input_ids_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)    # (batch_size, max_len)
        for i, ids in enumerate(batch_input_ids):
            input_ids_padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        
        return input_ids_padded, lengths

    def train_model(
        self, 
        dataset: Dataset,
        res_path: str,
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
            res_path: Path to save metrics CSV
        """
        logger.info(f"Training Neural HMM for {max_epochs} epochs")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Minibatch size: {minibatch_size}, Max inner loops: {max_inner_loops}")
        logger.info(f"Gradient clipping: {max_grad_norm}, Max sentence length: {max_sentence_length}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Metrics will be saved to: {res_path}")
        
        # Use Adam optimizer with specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # only sentences of 40 words or less
        filtered_dataset = []
        for example in dataset:
            forms = example["form"]
            if len(forms) > 0 and len(forms) <= max_sentence_length:
                filtered_dataset.append(example)
        logger.info(f"Filtered dataset: {len(filtered_dataset)} sentences (max length {max_sentence_length})")
        
        # Initialize metrics tracking
        epoch_metrics = []
        
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
                
                # Prepare batch: convert to padded tensors once
                input_ids_padded, lengths = self._pad_batch(batch)
                
                prev_log_prob = None
                # for inner_iter in tqdm(range(max_inner_loops), desc="  Inner loop", leave=False):
                for inner_iter in range(max_inner_loops):
                    optimizer.zero_grad()
                    
                    # Get matrices once for all sentences
                    log_T_matrix = self._get_transition_log_matrix()  # (num_states, num_states)   log_T_matrix[i, j] = log P(j | i) 
                    log_E_matrix = self._get_emission_log_matrix()    # (num_states, vocab_size)   log_E_matrix[s, v] = log P(v | s)
                    initial_log_probs = self._get_initial_log_probs()  # (num_states,)
                    
                    # Vectorised forward pass for entire batch
                    log_O_batch = self._forward_log_batched(
                        input_ids_padded, lengths, log_T_matrix, log_E_matrix, initial_log_probs
                    )  # (batch_size,) - log probability of each sequence
                    
                    # Compute total log probability and loss
                    total_log_prob = log_O_batch.sum()
                    batch_loss = -total_log_prob  # Negative log-likelihood
                    
                    # Average loss over batch
                    avg_batch_loss = batch_loss / len(batch)
                    
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
                
                total_loss += avg_batch_loss.detach()
                num_batches += 1
            
            average_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch+1}/{max_epochs}: Average loss = {average_loss:.4f}")
            
            # Compute and log metrics
            metrics = self.compute_metrics(dataset)
            metrics["epoch"] = epoch + 1
            metrics["avg_loss"] = average_loss.item() if isinstance(average_loss, torch.Tensor) else float(average_loss)
            epoch_metrics.append(metrics)
            
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} Metrics: "
                f"Normalized VI={metrics['normalized_vi']:.4f}, "
                f"Homogeneity={metrics['homogeneity']:.4f}, "
                f"Completeness={metrics['completeness']:.4f}, "
                f"V-score={metrics['v_score']:.4f}"
            )
        
        # Save metrics to CSV
        if epoch_metrics:
            fieldnames = [
                "normalized-VI",
                "homogeneity",
                "completeness",
                "V-score",
                "avg_loss",
                "epoch"
            ]
            with open(res_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(epoch_metrics)
            logger.info(f"Metrics saved to {res_path}")
        
        return epoch_metrics
    
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