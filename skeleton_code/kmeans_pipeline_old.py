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
from utils import calculate_entropy, calculate_mutual_information, calculate_variation_of_information, calculate_v_measure


# Set up logger
logger = logging.getLogger()

def generate_bert_embeddings(
    sentences: list, 
    word_embedding_path: str,
    batch_size: int = 32,
    model_name: str = "google-bert/bert-base-uncased",
    device: str = None
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Generate BERT embeddings for words from PTB sentences.

    Args:
        sentences: List of sentence dicts from PTB (each has "form" key with word list)
        word_embedding_path: Path to the word embedding file or cache
        batch_size: Batch size for processing unique words (default: 32)
        model_name: Name of the BERT model (default: "google-bert/bert-base-uncased")
        device: Device to use for the model (default: auto-detect)

    Returns:
        word_to_embedding: Dictionary mapping words to their BERT embeddings
        embeddings: Tensor of BERT embeddings
    
    Steps:
        1. Check if a cache file containing word embeddings exists on disk.
            a. If the cache exists, load the word_to_embedding dictionary from cache.
            b. If not, process sentences in batches to generate BERT embeddings for all unique words:
                i. 1. Load the BERT model and tokenizer.
                ii. Extract unique words from PTB sentence dicts
                iii. Tokenize each batch of unique words.
                iv. Pass each batch through the BERT model to get word embeddings.
                v. Store the embeddings in the word_to_embedding dictionary.
            c. Save the word_to_embedding dictionary to disk as a cache for future use.
        3. Return the word_to_embedding dictionary and stacked tensor of embeddings.
    """

    # Set the device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"Using device: {device}")

    # 1a. Check if a cache file containing word embeddings exists on disk.
    cache_path = word_embedding_path
    word_to_embedding = {}
    if cache_path and os.path.isfile(cache_path):
        logger.info(f"Loading word embeddings from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            word_to_embedding = pickle.load(f)
        logger.info(f"Cache loaded successfully. Found {len(word_to_embedding)} word embeddings.")
    
    # 1b. If not, generate BERT embeddings for all unique words
    else:
        logger.info("No cache found. Generating BERT embeddings for all unique words.")
        
        # ii. Load the BERT model and tokenizer.
        logger.info(f"Loading BERT model: {model_name}")
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # iii. Get all unique words from PTB sentence dicts
        logger.info("Getting all unique words from PTB sentence dicts")
        unique_words = set()
        for sentence in sentences:  # sentence is a dict with "form" key that contains list of word strings
            unique_words.update(sentence["form"])  # add all word strings into set of unique words
        unique_words = sorted(list(unique_words))
        logger.info(f"Found {len(unique_words)} unique words.")

        # iv. Process unique words in batches
        word_to_embedding = {}
        with torch.no_grad():   # Extracting embeddings so can speed up computation by not computing gradients

            total_unique_words = len(unique_words)

            with tqdm(total=total_unique_words, desc="Generating BERT embeddings") as pbar:
                for i in range(0, total_unique_words, batch_size):
                    batch_words = unique_words[i : i + batch_size]
                    # Tokenize batch
                    tokens = tokenizer(batch_words, padding=True, truncation=True, return_tensors="pt")
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    outputs = model(**tokens)

                    # Use subword pooling to get word embeddings:
                    # For each word, find its corresponding token indices, pool (mean) their embeddings
                    input_ids = tokens["input_ids"]  # (batch_size, seq_len)
                    attention_mask = tokens["attention_mask"]  # (batch_size, seq_len)
                    batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
                    for j, word in enumerate(batch_words):

                        # Remove special tokens ([CLS], [SEP], [PAD])
                        token_ids = input_ids[j]
                        mask = attention_mask[j].bool()
                        
                        # Filter out special tokens, get indices for real subword tokens
                        indices = []
                        for idx, tid in enumerate(token_ids):
                            token_str = batch_tokens[j][idx]
                            if token_str not in tokenizer.all_special_tokens and mask[idx]:
                                indices.append(idx)

                        # Pool (mean) the embeddings for subword tokens for this word
                        if len(indices) > 0:
                            subword_embs = outputs.last_hidden_state[j, indices, :]  # (num_subwords, hidden_dim)
                            word_emb = subword_embs.mean(dim=0)
                        else:
                            # Use [CLS] token embedding if no valid subword tokens found
                            logger.warning(f"No valid subword tokens for '{word}', using [CLS] token embedding.")
                            word_emb = outputs.last_hidden_state[j, 0, :]  # [CLS] token
                        
                        word_to_embedding[word] = word_emb.cpu()
                        pbar.update(1)
        # 2c. Save the word_to_embedding dictionary to disk as cache (using pickle for tensors)
        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(word_to_embedding, f)
                logger.info(f"Word embeddings cached to {cache_path}.")
            except Exception as e:
                logger.warning(f"Pickle cache failed: {e}. Not caching word embeddings.")

    # 3. Building embeddings for all unique words
    # Sorting for consistent ordering.
    word_list = sorted(word_to_embedding.keys())
    embeddings = torch.stack([word_to_embedding[w] for w in word_list])

    return word_to_embedding, embeddings


def train(
    word_to_embedding: dict[str, torch.Tensor], 
    num_clusters: int,
    save_path: str = None
) -> KMeansClassifier:
    """
    Train K-means classifier (pipeline function).
    
    Args:
        word_to_embedding: Dictionary mapping words to their BERT embeddings
        num_clusters: Number of clusters (matches number of POS tags)
        save_path: Path to save the model
    
    Returns:
        kmeans: Trained KMeansClassifier instance
    
    Steps:
        1. Creates KMeansClassifier instance
        2. Calls its train() method
        3. Handles saving
    """

    logger.info("Training K-means")

    # 1. Creates KMeansClassifier instance
    kmeans = KMeansClassifier(num_clusters=num_clusters, word_to_embedding=word_to_embedding)
    
    # 2. Calls its train() method that handles fitting
    kmeans.train()
    
    # 3. Handles saving
    if save_path is not None:
        logger.info(f"Saving K-means model to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(kmeans, f)
    else:
        logger.warning("No save path provided. K-means model not saved")
    
    logger.info("K-means training successful")

    return kmeans


def eval(
    dataset_split: Dataset,
    kmeans: KMeansClassifier = None,
    load_path: str = None,
    res_path: str = "kmeans_result.csv"
):
    """

    Evaluate a trained K-means model on the specified dataset split.
    Writes results to res_path.

    Args:
        dataset_split: Dataset split to evaluate on
        kmeans: Trained KMeansClassifier instance
        load_path: Path to load the model from
        res_path: Path to save the results to

    Returns:
        Dictionary containing the evaluation metrics

    Steps:
        1. For each sentence in the evaluation dataset:
           a. Predict cluster labels for each word using the model
           b. Collect gold POS tags for each word
        2. Compare predicted cluster labels with gold POS tags across all sentences
        3. Compute evaluation metrics (Entropy, Mutual Information, Variation of Information, V-measure)
        4. Save detailed predictions and computed metrics to `res_path`
    """

    logger.info("Evaluating K-means")

    # Load the model from load_path if kmeans is not provided
    if kmeans is None:
        if load_path is None:
            raise ValueError(
                "At least one of K-means model and load_path should be provided"
            )
        # Load K-means model
        logger.info(f"Loading K-means model from {load_path}")
        with open(load_path, "rb") as f:
            kmeans = pickle.load(f)

    # Evaluate
    num_samples = len(dataset_split)
    results = []
    all_true_tags = []
    all_pred_tags = []

    for i, example in enumerate(tqdm(dataset_split, "K-means testing", num_samples)):
        # Predicting and getting gold POS tags for each word
        forms = example["form"]  # List of word strings
        true_tags = example["tags"]  # List of tag integers
        pred_tags = kmeans.inference(forms)  # List of cluster integers

        sentence = " ".join(forms)

        # Compute per-example V-measure, VI, MI, entropy
        homo_score, comp_score, v_score = calculate_v_measure(true_tags, pred_tags)
        vi, normalized_vi = calculate_variation_of_information(true_tags, pred_tags)
        mi = calculate_mutual_information(true_tags, pred_tags)
        entropy_true = calculate_entropy(true_tags)
        entropy_pred = calculate_entropy(pred_tags)

        results.append(
            [
                i + 1,
                sentence,
                vi,
                normalized_vi,
                homo_score,
                comp_score,
                v_score,
                mi,
                entropy_true,
                entropy_pred
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

    logger.info("Computing whole-dataset mutual information")
    mi_whole = calculate_mutual_information(all_true_tags, all_pred_tags)
    
    logger.info("Computing whole-dataset entropies")
    entropy_true_whole = calculate_entropy(all_true_tags)
    entropy_pred_whole = calculate_entropy(all_pred_tags)

    print(f"Homogeneity score: {homo_score_whole}")
    print(f"Completeness score: {comp_score_whole}")
    print(f"V-measure: {v_score_whole}")
    print(f"Variation of information: {vi_whole}")
    print(f"Normalized VI: {normalized_vi_whole}")
    print(f"Mutual information: {mi_whole}")
    print(f"Entropy (gold): {entropy_true_whole}")
    print(f"Entropy (pred): {entropy_pred_whole}")

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
                "V-score",
                "mutual-information",
                "entropy-gold",
                "entropy-pred"
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
                v_score_whole,
                mi_whole,
                entropy_true_whole,
                entropy_pred_whole,
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

    # 1. Load PTB dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    # 2. Generate/load BERT embeddings for all unique words
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_to_embedding, _ = generate_bert_embeddings(
        sentences,
        word_embedding_path=word_embedding_path,
        batch_size=32,
        model_name="google-bert/bert-base-uncased",
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
    
    # 4. Train K-means model
    with torch.no_grad():
        kmeans = train(
            word_to_embedding=word_to_embedding,
            num_clusters=len(tag_mapping),
            save_path=save_path,
        )

        # 5. Evaluate K-means model
        eval(
            dataset_splits["test"],
            kmeans=kmeans,
            res_path=res_path,
        )