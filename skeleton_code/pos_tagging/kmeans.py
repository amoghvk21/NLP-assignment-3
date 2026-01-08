from pos_tagging.base import BaseUnsupervisedClassifier
from datasets import Dataset
from typing import List, Optional
import torch
import logging
from sklearn.cluster import KMeans
import numpy as np
import warnings
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Ignore sklean warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set up logger
logger = logging.getLogger()

class KMeansClassifier(BaseUnsupervisedClassifier):
    def __init__(self, num_clusters, device: torch.device, bert_model: AutoModel, tokenizer: AutoTokenizer):
        """
        Initialize K-means classifier
        
        Args:
            num_clusters: Number of clusters (should match number of POS tags)
            device: Device to run model on
            bert_model: Pre-loaded BERT model
            tokenizer: Pre-loaded BERT tokenizer
        
        Store: 
            - num_clusters (int)
            - device: Device to run model on
            - kmeans_model (sklearn.cluster.KMeans) - will be set in train()
            - bert_model: BERT model reused for inference
            - tokenizer: BERT tokenizer reused for inference
        """
        self.num_clusters = num_clusters
        self.device = device
        self.kmeans_model = None  # Will be set in train()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
    
    def train(self, embeddings_tensor: torch.Tensor) -> None:
        """
        Train (fit) K-means clustering on contextual word embeddings.
        
        Clusters all word tokens (instances) from all sentences
        Each word occurrence gets its own embedding based on context.

        Args:
            embeddings_tensor: Tensor of shape (total_word_tokens, hidden_dim) containing all contextual embeddings from training sentences

        Steps:
            1. Convert embeddings tensor to numpy array
            2. Fit sklearn KMeans model on all word token embeddings
        """
        logger.info("Fitting K-means model on contextual word embeddings")
        logger.info(f"Total word tokens to cluster: {embeddings_tensor.shape[0]}")
        
        # 1. Convert embeddings tensor to numpy array on CPU
        # embeddings_tensor: (total_words, hidden_dim)
        embeddings_np = embeddings_tensor.detach().cpu().numpy()
        
        # 2. Fit sklearn KMeans model on all word token embeddings
        # This clusters word instances, not word types
        logger.info("Starting KMeans fit")
        self.kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(embeddings_np)
        logger.info("KMeans fit complete.")
        
        logger.info(f"K-means fitting complete. {len(cluster_labels)} word tokens clustered into {self.num_clusters} clusters.")
    
    def inference(self, word_list: List[str]) -> List[int]:
        """
        Map a sequence of words to cluster labels.
        For each sentence, generate contextual embeddings and predict clusters (same word can get different embedding based on context)

        Args:
            word_list: List of word strings in a sentence
        
        Returns:
            List of cluster labels (integers) for each word in word_list
        
        Steps:
            1. Tokenise the sentence with word alignment (is_split_into_words=True)
            2. Pass through BERT to get contextualised embeddings
            3. For each word, align it to its subword tokens using word_ids()
            4. Mean pool subword token embeddings to get word embedding
            5. Predict cluster for each word embedding using trained K-means model
            6. Return list of cluster labels
        """
        
        # Handle empty word list
        if len(word_list) == 0:
            return []
        
        # 1. Tokenise the sentence with word alignment enabled
        # This allows us to map tokens back to words using word_ids()
        # We need to encode first to get word_ids, then convert to tensors
        encoded = self.tokenizer(
            word_list,
            is_split_into_words=True,  # Important: tells tokenizer words are pre-split
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        tokens = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 2. Pass through BERT to get contextualised embeddings
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
            hidden_states = outputs.last_hidden_state[0]  # (seq_len, hidden_dim); index 0 to get first (and only) sentence

        # 3. For each word, align it to its subword tokens using word_ids()
        word_ids = encoded.word_ids()  # (seq_len,)
        
        # 4. Mean pool subword token embeddings to get word embedding
        word_embeddings = []
        current_word_idx = None
        subword_indices = []
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD]) - skip
                continue
            
            # We have moved to a new word
            if word_idx != current_word_idx:
                # Check if there are any subwords to pool for previous word
                if current_word_idx is not None and len(subword_indices) > 0:
                    # Get all the subword embeddings
                    subword_embeddings = hidden_states[subword_indices, :]  # (num_subwords, hidden_dim)

                    # Mean pool the subword embeddings
                    word_embedding = subword_embeddings.mean(dim=0)  # (hidden_dim,)

                    # Add the pooled word embedding to the result list
                    word_embeddings.append(word_embedding)
                
                # Start collecting subwords for new word we are at now
                current_word_idx = word_idx
                subword_indices = [token_idx]

            # We are still on same word. Add this subword token to the list
            else:
                subword_indices.append(token_idx)
        
        # Process last word
        if current_word_idx is not None and len(subword_indices) > 0:
            # Get all the subword embeddings
            subword_embeddings = hidden_states[subword_indices, :]
            # Mean pool the subword embeddings
            word_embedding = subword_embeddings.mean(dim=0)
            # Add the pooled word embedding to the result list
            word_embeddings.append(word_embedding)
        
        # Handle edge cases: mismatch between words and embeddings
        if len(word_embeddings) != len(word_list):
            raise RuntimeError(
                f"Inference: Expected {len(word_list)} word embeddings, "
                f"got {len(word_embeddings)}. Mismatch between words and embeddings."
            )
            # This deals with fixing this error TODO remove if not needed
            """
            logger.warning(
                f"Inference: Expected {len(word_list)} word embeddings,", 
                f"got {len(word_embeddings)}. Padding/truncating."
            )

            # word_embeddings is shorter than the number of words
            # Pad word_embeddings with the last embedding
            if len(word_embeddings) < len(word_list):
                # Repeat last embedding for missing words
                last_emb = word_embeddings[-1] if word_embeddings else hidden_states[0, :]
                word_embeddings.extend([last_emb] * (len(word_list) - len(word_embeddings)))

            # word_embeddings is longer than the number of words
            # Shorten word_embeddings to the number of words
            else:
                word_embeddings = word_embeddings[:len(word_list)]
            """
        
        # 5. Predict cluster for each word embedding using trained K-means model
        if self.kmeans_model is None:
            raise RuntimeError("K-means model has not been trained. Call train() before inference.")
        
        if len(word_embeddings) > 0:
            # Stack embeddings and convert to numpy
            word_embs_tensor = torch.stack(word_embeddings).cpu().numpy()  # (num_words, hidden_dim)
            # Predict clusters
            pred_tags = self.kmeans_model.predict(word_embs_tensor).tolist()
        else:
            raise RuntimeError("No embeddings generated for inference; cannot assign clusters for input words. ")
        
        # 6. Return list of cluster labels
        return pred_tags