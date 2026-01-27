import os
import torch
from datasets import DatasetDict, Dataset
from logging_nlp import get_logger
from tqdm import tqdm
import csv

from pos_tagging.nhmm import NeuralHMMClassifier
from preprocess_dataset import load_ptb_dataset, wrap_dataset, create_tag_mapping
from utils import calculate_v_measure, calculate_variation_of_information


logger = get_logger(__name__)


def train(
    dataset_splits: DatasetDict,
    num_states: int,
    vocab: list,
    tag_mapping: dict,
    device: torch.device,
    max_epochs: int = 50,
    save_path: str = None,
    res_path: str = "nhmm_training_metrics.csv"
):
    """
    Train Neural HMM model.
    
    Args:
        dataset_splits: Dataset splits
        num_states: Number of hidden states (POS tags)
        vocab: List of sentences for building vocabulary
        tag_mapping: Dictionary mapping tag strings to integers
        device: Device to run model on
        max_epochs: Maximum number of training epochs
        save_path: Path to save the model
        res_path: Path to save per-epoch training metrics
    """
    logger.info("Training Neural HMM")
    
    # Initialize model
    model = NeuralHMMClassifier(
        num_states=num_states,
        vocab=vocab,
        tag_mapping=tag_mapping,
        device=device
    )
    
    # Train model    
    model.train_model(dataset_splits["train"], res_path=res_path, max_epochs=max_epochs)
    
    # Save model
    if save_path is not None:
        logger.info(f"Saving Neural HMM model to {save_path}")
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    else:
        logger.warning("No save path provided. Neural HMM model not saved")
    
    logger.info("Neural HMM training complete.")
    return model

def eval(
    dataset_split: Dataset,
    model: NeuralHMMClassifier,
    res_path: str = "nhmm_result.csv"
):
    """
    Evaluate a trained NHMM model on the specified dataset split.
    Saves results to res_path.

    Args:
        dataset_split: Dataset split to evaluate on
        model: Trained NeuralHMM model
        res_path: Path to save the results to
    """

    logger.info("Evaluating NHMM")

    num_samples = len(dataset_split)
    results = []
    all_true_tags = []
    all_pred_tags = []

    for i, example in enumerate(tqdm(dataset_split, desc="NHMM testing", total=num_samples)):
        # Predicting and getting gold POS tags for each word
        forms = example["form"]  # List of word strings
        true_tags = example["tags"]  # List of tag integers
        
        # Skip empty sentences
        if len(forms) == 0:
            continue
        
        # Predict tags using model inference
        pred_tags = model.inference(forms)
        
        # # Ensure same length
        # min_len = min(len(true_tags), len(pred_tags))
        # true_tags = true_tags[:min_len]
        # pred_tags = pred_tags[:min_len]

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
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(res_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
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


def train_and_test(
    tag_name: str,
    subset: int = None,
    max_epochs: list = [50],
    load_path: str = None,
    save_path: str = None,
    res_path: str = None,
):
    """
    Train and test NeuralHMM on the specified dataset.

    Args:
        tag_name: "upos" or "xpos"
        subset: How many examples to use (None = all)
        max_epochs: List with [max_epochs, ...] values
        load_path: Path to load model (optional)
        save_path: Path to save model
        res_path: Path to save results
    """

    logger.info("Training and testing NeuralHMM")
    logger.warning(f"Using {tag_name} as tag")

    if save_path is None:
        raise ValueError("save_path must be provided for training and evaluation")

    # Load dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    # Create tag mapping
    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set)
    }[tag_name]

    def map_tags(examples):
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tags, desc="Mapping tags")

    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Train NHMM model
    model = train(
        dataset_splits=dataset_splits,
        num_states=len(tag_mapping),
        vocab=sentences,
        tag_mapping=tag_mapping,
        device=device,
        max_epochs=max_epochs[0] if isinstance(max_epochs, list) else max_epochs,
        save_path=save_path,
        res_path=res_path
    )

    # Evaluate NHMM model
    model.eval()
    with torch.no_grad():
        test_res_path = f"{res_path}_all_test.csv"
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(test_res_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        eval(
            dataset_splits["test"],
            model=model,
            res_path=test_res_path,
        )

    logger.info("NeuralHMM training/testing complete.")

def test(
    tag_name: str,
    subset: int = None,
    load_path: str = None,
    res_path: str = None,
):
    """
    Test NeuralHMM on the specified dataset.

    Args:
        tag_name: "upos" or "xpos"
        subset: How many examples to use (None = all)
        load_path: Path to load trained model
        res_path: Path to save results
    """
    logger.info("Testing NeuralHMM")
    logger.warning(f"Using {tag_name} as tag")

    if load_path is None:
        raise ValueError("load_path must be provided for testing")

    # Load dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    # Create tag mapping
    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set)
    }[tag_name]

    def map_tags(examples):
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tags, desc="Mapping tags")

    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")
    
    # Reconstruct model for evaluation
    model = NeuralHMMClassifier(
        num_states=len(tag_mapping),
        vocab=sentences,
        tag_mapping=tag_mapping,
        device=device
    )
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    # Call eval() to do the actual evaluation
    with torch.no_grad():
        eval(
            dataset_splits["test"],
            model=model,
            res_path=res_path,
        )
    
    logger.info("NeuralHMM testing complete.")
