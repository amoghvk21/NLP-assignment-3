from pos_tagging.base import BaseUnsupervisedClassifier
from datasets import Dataset
from typing import List
import torch
import logging
from sklearn.cluster import KMeans
import numpy as np
import warnings
from tqdm import tqdm
from collections import Counter


# Ignore sklean warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set up logger
logger = logging.getLogger()

class KMeansClassifier(BaseUnsupervisedClassifier):
    def __init__(self, num_clusters, word_to_embedding):
        """
        Initialize K-means classifier.
        
        Args:
            num_clusters: Number of clusters (should match number of POS tags)
            word_to_embedding: Dictionary mapping words (strings) to their BERT embeddings (torch.Tensor)
        
        Store: 
            - num_clusters (int)
            - word_to_embedding (dict: word -> torch.Tensor)
            - kmeans_model (sklearn.cluster.KMeans) - will be set in train()
            - word_to_cluster mapping (dict: word -> int) - will be created in train()
            - most_frequent_cluster (int) - will be set in train() for OOV handling
        """
        self.num_clusters = num_clusters
        self.word_to_embedding = word_to_embedding
        self.kmeans_model = None  # Will be set in train()
        self.word_to_cluster = {}  # Will be created in train()
        self.most_frequent_cluster = 0  # Will be set in train() for OOV handling
    
    def train(self) -> dict[str, int]:
        """
        Train (fit) K-means clustering on word embeddings.
        
        K-Means fitting happens here. Called from the pipeline function

        Returns:
            word_to_cluster: Dictionary mapping words to cluster labels

        Steps:
            1. Extract all words and their embeddings
            2. Convert embeddings to numpy array
            3. Fit sklearn KMeans model
            4. Create word_to_cluster mapping
        """
        logger.info("Fitting K-means model on word embeddings")
        
        # 1. Extract all words and their embeddings
        words = sorted(self.word_to_embedding.keys())

        # 2. Convert embeddings to numpy array with progress bar
        embeddings = []
        for w in tqdm(words, desc="Stacking embeddings"):
            embeddings.append(self.word_to_embedding[w])
        embeddings = torch.stack(embeddings).numpy()
        
        # 3. Fit sklearn KMeans model
        # Fitting itself does not support explicit progress bars, so we log progress before and after.
        logger.info("Starting KMeans fit")
        self.kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(embeddings)
        logger.info("KMeans fit complete.")

        # 4. Create word_to_cluster mapping with progress bar
        self.word_to_cluster = {}
        for i, word in enumerate(tqdm(words, desc="Assigning clusters")):
            self.word_to_cluster[word] = int(cluster_labels[i])
        
        # Calculate most frequent cluster for OOV handling
        cluster_counts = Counter(cluster_labels)
        self.most_frequent_cluster = int(cluster_counts.most_common(1)[0][0])
        logger.info(f"Most frequent cluster: {self.most_frequent_cluster} (for OOV handling)")
        
        logger.info(f"K-means fitting complete. {len(self.word_to_cluster)} words clustered into {self.num_clusters} clusters.")
        
        return self.word_to_cluster
    
    def inference(self, word_list):
        """
        Map a sequence of words to its cluster label.
        
        Args:
            word_list: List of word strings
        
        Returns:
            List of cluster labels (integers) for each word in word_list
        
        Steps:
            1. For each word in word_list:
                - Look up its cluster in word_to_cluster
                - Handle unknown words (OOV)
            2. Return list of cluster labels
        """
        
        # logger.info("Mapping words to cluster labels")
        
        # 1. For each word in word_list:
        pred_tags = []
        oov_count = 0
        for word in word_list:
            # 1.a. Look up its cluster in word_to_cluster
            if word in self.word_to_cluster:
                pred_tags.append(self.word_to_cluster[word])
            # 1.b. If the word is not in word_to_cluster, assign it to most frequent cluster
            else:
                oov_count += 1
                pred_tags.append(self.most_frequent_cluster)
                logger.warning(f"Unknown word '{word}' not in word_to_cluster")
        
        if oov_count > 0:
            logger.warning(f"Encountered {oov_count} unknown word(s), assigned to most frequent cluster {self.most_frequent_cluster}")
        
        # logger.info("Word mapping complete")
        
        # 2. Return list of cluster labels
        return pred_tags