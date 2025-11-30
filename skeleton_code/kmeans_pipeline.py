from transformers import AutoModel, AutoTokenizer
import torch
import logging
import csv
from datasets import Dataset
import os
import pickle
from sklearn.cluster import KMeans
from datasets import DatasetDict
from tqdm import tqdm

from pos_tagging.kmeans import KMeansClassifier
from preprocess_dataset import *
from utils import calculate_variation_of_information, calculate_v_measure


# Set up logger
logger = logging.getLogger()

BERT_MODEL_NAME = "google-bert/bert-base-uncased"

def generate_bert_embeddings(
    sentences: list, 
    word_embedding_path: str,
    batch_size: int = 32,
    device: str = None
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Generate context-dependent BERT embeddings for words from PTB sentences.
    
    Each word gets a contextual embedding based on its sentence context.
    The same word type can have different embeddings in different sentences.

    Args:
        sentences: List of sentence dicts from PTB (each has "form" key with word list)
        word_embedding_path: Path to the word embedding cache file
        batch_size: Batch size for processing sentences (default: 32)
        device: Device to use for the model (default: auto-detect)

    Returns:
        all_word_embeddings: List of tensors, each containing word embeddings for one sentence
        embeddings_tensor: Stacked tensor of all word embeddings (for clustering)
    
    Steps:
        1. Check if a cache file containing word embeddings exists on disk.
            a. If the cache exists, load the all_word_embeddings list from cache.
            b. If not, process sentences in batches to generate BERT embeddings:
                i. Load the BERT model and tokenizer.
                ii. For each sentence batch:
                    - Tokenize sentences with word alignment (is_split_into_words=True)
                    - Pass through BERT to get contextualized hidden states
                    - For each word, align it to its subword tokens using word_ids()
                    - Pool (mean) the subword token embeddings to get word embedding
                    - Store contextualized word embeddings for every word occurrence
            c. Save the all_word_embeddings list to disk as a cache for future use.
        2. Flatten all sentence embeddings into a single tensor for clustering.
        3. Return embeddings list and flattened tensor.
    """

    # Set the device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"Using device: {device}")

    all_word_embeddings = []

    # 1a. Check if a cache file containing word embeddings exists on disk.
    if word_embedding_path and os.path.isfile(word_embedding_path):
        logger.info(f"Loading contextual embeddings from cache: {word_embedding_path}")
        with open(word_embedding_path, "rb") as f:
            all_word_embeddings = pickle.load(f)
        logger.info(f"Cache loaded successfully. Found {len(all_word_embeddings)} sentences.")
    
    # 1b. If not cached, generate BERT embeddings for all sentences
    if not all_word_embeddings:
        logger.info("No cache found. Generating context-dependent BERT embeddings.")
        
        # i. Load the BERT model and tokeniser
        logger.info(f"Loading BERT model: {BERT_MODEL_NAME}")
        model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model.eval()
        
        # Stores embeddings split by sentence
        all_word_embeddings = []
        
        # ii. Process sentences in batches
        total_num_sentences = len(sentences)
        with torch.no_grad():
            with tqdm(total=total_num_sentences, desc="Generating contextual embeddings") as pbar:
                for i in range(0, total_num_sentences, batch_size):
                    batch_sentences = sentences[i : i + batch_size]
                    
                    # Extract actual words from sentence
                    batch_words = [s["form"] for s in batch_sentences]
                    
                    # Tokenise and convert to tensors
                    encoded = tokenizer(
                        batch_words,
                        is_split_into_words=True,  # Allows us to align tokens to words easier later
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    tokens = {k: v.to(device) for k, v in encoded.items()}
                    
                    # Returns a list (batch_size, seq_len) mapping each token to its word index (None for special tokens)
                    # Iterates through each sentence in the batch
                    batch_word_ids = [encoded.word_ids(batch_index=i) for i in range(len(batch_words))]   # TODO: Check
                    
                    # Pass through BERT to get contextualised embeddings
                    outputs = model(**tokens)
                    hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                    
                    # For each sentence in batch, extract word embeddings
                    for j, sentence_dict in enumerate(batch_sentences): # TODO: Check if i could have iterated through batch_words instead
                        words = sentence_dict["form"]
                        word_embeddings = [] # Will store embeddings for each word in the sentence
                        
                        # Get the ids mapping each token to its word index
                        word_ids = batch_word_ids[j] # (seq_len,)
                        
                        # Group tokens by word and pool their embeddings
                        current_word_idx = None # Will store the index of the current word
                        subword_indices = [] # Will store the indices of the subwords of the current word
                        
                        # Iterate over each token in the sentence
                        # Either append subword
                        # Or we are on new word and max pool what we have collected so far
                        for token_idx, word_idx in enumerate(word_ids):
                            if word_idx is None:
                                # Special token ([CLS], [SEP], [PAD]) - skip
                                continue
                            
                            # We have moved to a new word
                            if word_idx != current_word_idx:
                                # Check if there are any subwords to pool for previous word
                                if current_word_idx is not None and len(subword_indices) > 0:
                                    # Get all the subword embeddings
                                    subword_embeddings = hidden_states[j, subword_indices, :]  # (num_subwords, hidden_dim)
                                    
                                    # Pool the subword embeddings
                                    word_embedding = subword_embeddings.mean(dim=0)  # (hidden_dim,)
                                    
                                    # Add the pooled embedding to the result list
                                    word_embeddings.append(word_embedding)
                                
                                # Start collecting subwords for the new word we are at now
                                current_word_idx = word_idx
                                subword_indices = [token_idx]
                            
                            # We are still on same word. Add this subword token to the list
                            else:
                                subword_indices.append(token_idx)
                        
                        # Process last word
                        if current_word_idx is not None and len(subword_indices) > 0:
                            # Get all the subword embeddings
                            subword_embeddings = hidden_states[j, subword_indices, :]
                            # Pool the subword embeddings
                            word_embedding = subword_embeddings.mean(dim=0)
                            # Add the pooled embedding to the result list
                            word_embeddings.append(word_embedding)
                        
                        # Verify we got embeddings for all words
                        if len(word_embeddings) != len(words):
                            raise RuntimeError(
                                f"Sentence {i+j}: Expected {len(words)} word embeddings,", 
                                f"got {len(word_embeddings)}. Padding/truncating."
                            )
                            # This deals with fixing this error 
                            """
                            logger.warning(
                                f"Sentence {i+j}: Expected {len(words)} word embeddings,", 
                                f"got {len(word_embeddings)}. Padding/truncating."
                            )
                            
                            # word_embeddings is shorter than the number of words
                            # Pad word_embeddings with the last embedding
                            if len(word_embeddings) < len(words):
                                # Repeat last embedding for missing words
                                last_emb = word_embeddings[-1] if word_embeddings else hidden_states[j, 0, :]
                                word_embeddings.extend([last_emb] * (len(words) - len(word_embeddings)))
                            
                            # word_embeddings is longer than the number of words
                            # Shorten word_embeddings to the number of words
                            else:
                                word_embeddings = word_embeddings[:len(words)]
                            """
                        
                        # Store embeddings for this sentence
                        all_word_embeddings.append(torch.stack(word_embeddings))
                        pbar.update(1)
        
        # 1c. Save the all_word_embeddings list to cache
        if word_embedding_path:
            try:
                with open(word_embedding_path, "wb") as f:
                    pickle.dump(all_word_embeddings, f)
                logger.info(f"Contextual embeddings cached to {word_embedding_path}.")
            except Exception as e:
                logger.warning(f"Pickle cache failed: {e}. Not caching word embeddings to {word_embedding_path}.")

    # 2. Flatten all sentence embeddings into a single tensor for clustering
    # Each element in all_word_embeddings is (sentence_len, hidden_dim)
    if all_word_embeddings:
        embeddings_tensor = torch.cat(all_word_embeddings, dim=0)  # flatten into (total_words, hidden_dim)
        logger.info(f"Total word tokens: {embeddings_tensor.shape[0]}, embedding dim: {embeddings_tensor.shape[1]}")
    else:
        # Create empty 2D tensor with shape (0, hidden_dim)
        embeddings_tensor = torch.empty((0, 768))
        logger.warning("No embeddings generated!")

    # 3. Return embeddings list and flattened tensor
    return all_word_embeddings, embeddings_tensor


def train(
    embeddings_tensor: torch.Tensor,
    num_clusters: int,
    device: torch.device,
    save_path: str = None
) -> None:
    """
    Train K-means classifier on contextual word embeddings (pipeline function).
    
    Args:
        embeddings_tensor: Tensor of shape (total_word_tokens, hidden_dim) containing all contextual embeddings
        num_clusters: Number of clusters (matches number of POS tags)
        device: Device to run model on
        save_path: Path to save the model

    Steps:
        1. Creates KMeansClassifier instance (model/tokenizer loaded internally)
        2. Calls its train() method with embeddings tensor
        3. Handles saving
    """

    logger.info("Training K-means on contextual embeddings")

    # 1. Creates KMeansClassifier instance
    kmeans = KMeansClassifier(
        num_clusters=num_clusters,
        device=device
    )
    
    # 2. Calls its train() method that handles fitting
    kmeans.train(embeddings_tensor)
    
    # 3. Saves the model
    if save_path is not None:
        logger.info(f"Saving K-means model to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(kmeans, f)
    else:
        logger.warning("No save path provided. K-means model not saved")
    
    logger.info("K-means training successful")


def eval(
    dataset_split: Dataset,
    load_path: str,
    res_path: str = "kmeans_result.csv"
):
    """

    Evaluate a trained K-means model on the specified dataset split.
    Writes results to res_path.

    Args:
        dataset_split: Dataset split to evaluate on
        load_path: Path to load the model from
        res_path: Path to save the results to

    Returns:
        Dictionary containing the evaluation metrics

    Steps:
        1. For each sentence in the evaluation dataset:
           a. Predict cluster labels for each word using the model
           b. Collect gold POS tags for each word
        2. Compare predicted cluster labels with gold POS tags across all sentences
        3. Compute evaluation metrics (Variation of Information, V-measure)
        4. Save detailed predictions and computed metrics to `res_path`
    """

    logger.info("Evaluating K-means")

    # Load K-means model from load_path
    if load_path is None:
        raise ValueError("load_path must be provided for evaluation")
            
    logger.info(f"Loading K-means model from {load_path}")
    with open(load_path, "rb") as f:
        kmeans = pickle.load(f)

    # Evaluate
    num_samples = len(dataset_split)
    results = []
    all_true_tags = []
    all_pred_tags = []

    for i, example in enumerate(tqdm(dataset_split, desc="K-means testing", total=num_samples)):
        # Predicting and getting gold POS tags for each word
        forms = example["form"]  # List of word strings
        true_tags = example["tags"]  # List of tag integers
        pred_tags = kmeans.inference(forms)  # List of cluster integers

        sentence = " ".join(forms)

        # Compute per-example V-measure and VI
        homo_score, comp_score, v_score = calculate_v_measure(true_tags, pred_tags)
        vi, normalized_vi = calculate_variation_of_information(true_tags, pred_tags)

        results.append(
            [
                i + 1,
                sentence,
                vi,
                normalized_vi,
                homo_score,
                comp_score,
                v_score
            ]
        )

        # Append all these tags to compute whole dataset metrics
        all_true_tags.extend(true_tags)
        all_pred_tags.extend(pred_tags)

    # Compute whole-dataset metrics
    logger.info("Computing whole-dataset V-measure")
    homo_score_whole, comp_score_whole, v_score_whole = calculate_v_measure(
        all_true_tags, all_pred_tags
    )

    logger.info("Computing whole-dataset VI")
    vi_whole, normalized_vi_whole = calculate_variation_of_information(
        all_true_tags, all_pred_tags
    )

    print(f"Homogeneity score: {homo_score_whole}")
    print(f"Completeness score: {comp_score_whole}")
    print(f"V-measure: {v_score_whole}")
    print(f"Variation of information: {vi_whole}")
    print(f"Normalized VI: {normalized_vi_whole}")

    # Save results to CSV
    logger.info(f"Saving results to {res_path}")
    with open(res_path, "w+", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "sentence",
                "VI",
                "normalized-VI",
                "homogeneity",
                "completeness",
                "V-score"
            ]
        )
        # Save whole-dataset results as id==0
        writer.writerow(
            [
                0,
                "-",
                vi_whole,
                normalized_vi_whole,
                homo_score_whole,
                comp_score_whole,
                v_score_whole
            ]
        )
        # Save per-example results
        writer.writerows(results)


def test(
    tag_name: str,
    subset: int = None,
    load_path: str = None,
    res_path: str = "kmeans_result.csv"
):
    """
    Test or evaluate a trained K-means model on the specified dataset split.

    Args:
        tag_name: "upos" or "xpos" (which POS tag type to use for evaluation)
        subset: How many examples to use for testing (None = use all)
        load_path: Path to the saved K-means model (.pkl)
        res_path: Path to save the results (e.g., CSV file)

    Returns:
        None (writes results/metrics to res_path)

    Steps:
        1. Load dataset split (e.g., PTB validation or test set)

    Notes:
        - The function assumes all necessary data and the trained model have already been generated/saved.
        - This does _not_ retrain K-means; it only loads and applies an existing model.
    """
    
    logger.info("Testing K-means")

    logger.warning(f"Using {tag_name} as tag")

    # 1. Load dataset split (e.g., PTB validation or test set)
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    # 2. Create tag mapping
    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]
    
    def map_tags(examples):
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples
    
    dataset = dataset.map(map_tags, desc="Mapping tags")

    dataset_splits = DatasetDict({"train": dataset, "test": dataset})
    
    # 3. Call eval() to do the actual evaluation
    with torch.no_grad():
        eval(
            dataset_splits["test"],
            load_path=load_path,
            res_path=res_path,
        )

    logger.info("K-means testing successful")

def train_and_test(
    tag_name: str,
    subset: int = None,
    word_embedding_path: str = None,
    save_path: str = None,
    res_path: str = None
):
    """
    Train and test a K-means model on the specified dataset split.

    Args:
        tag_name: "upos" or "xpos" (which POS tag type to use for evaluation)
        subset: How many examples to use (None = use all)
        word_embedding_path: Path to load/save BERT embeddings cache (.pt)
        save_path: Path to save the K-means model (.pkl)
        res_path: Path to save the results CSV

    Steps:
        1. Load PTB dataset
        2. Generate/load BERT embeddings for all unique words
        3. Create tag mapping
        4. Train K-means model
        5. Evaluate K-means model
    """

    logger.info("Training and testing K-means")
    logger.warning(f"Using {tag_name} as tag")

    if save_path is None:
        raise ValueError("save_path must be provided for training and evaluation")

    # 1. Load PTB dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    # 2. Generate/load context-dependent BERT embeddings for all sentences
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_word_embeddings, embeddings_tensor = generate_bert_embeddings(
        sentences,
        word_embedding_path=word_embedding_path,
        batch_size=32,
        device=device
    )

    # 3. Create tag mapping
    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]
    
    def map_tags(examples):
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples
    
    dataset = dataset.map(map_tags, desc="Mapping tags")
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})
    
    with torch.no_grad():
        # 4. Train K-means model on contextual embeddings
        train(
            embeddings_tensor=embeddings_tensor,
            num_clusters=len(tag_mapping),
            device=device,
            save_path=save_path,
        )

        # 5. Evaluate K-means model
        eval(
            dataset_splits["test"],
            load_path=save_path,
            res_path=res_path,
        )