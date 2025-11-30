import torch
from datasets import DatasetDict
from logging_nlp import get_logger

from pos_tagging.hmm import NeuralHMM
from utils import load_ptb_dataset, wrap_dataset, create_tag_mapping

logger = get_logger(__name__)

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
        tag_name: "upos" or "xpos" for POS tag type
        subset: How many examples to use (None = all)
        max_epochs: List with [max_epochs, ...] values for training epochs
        load_path: Path to load model (optional)
        save_path: Path to save model (.pt)
        res_path: Path to save results csv

    Steps:
        1. Load PTB dataset
        2. Create tag mapping
        3. Train NHMM model
        4. Evaluate NHMM model
    """
    logger.info("Training and testing NeuralHMM")
    logger.warning(f"Using {tag_name} as tag")

    if save_path is None:
        raise ValueError("save_path must be provided for training and evaluation")

    # 1. Load PTB dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    # 2. Create tag mapping
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # 3. Train NHMM model
        model = NeuralHMM(num_states=len(tag_mapping), vocab=sentences, tag_mapping=tag_mapping, device=device)
        model.train_model(dataset_splits["train"], max_epochs=max_epochs)
        torch.save(model.state_dict(), save_path)

        # 4. Evaluate NHMM model
        model.eval()
        results = model.evaluate(dataset_splits["test"])
        if res_path is not None:
            # Save the results to CSV (results should be in DataFrame-like or list-of-dict form)
            import pandas as pd
            pd.DataFrame(results).to_csv(res_path, index=False)

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
        tag_name: "upos" or "xpos" for POS tag type
        subset: How many examples to use (None = all)
        load_path: Path to load trained model (.pt)
        res_path: Path to save results csv
    """
    logger.info("Testing NeuralHMM")
    logger.warning(f"Using {tag_name} as tag")

    if load_path is None:
        raise ValueError("load_path must be provided for testing")

    # 1. Load PTB dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    # 2. Create tag mapping
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # 3. Load NHMM model
        model = NeuralHMM(num_states=len(tag_mapping), vocab=sentences, tag_mapping=tag_mapping, device=device)
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.eval()

        # 4. Evaluate NHMM model
        results = model.evaluate(dataset_splits["test"])
        if res_path is not None:
            import pandas as pd
            pd.DataFrame(results).to_csv(res_path, index=False)
    
    logger.info("NeuralHMM testing complete.")
