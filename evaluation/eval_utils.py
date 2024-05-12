import re
import time
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from num2words import num2words
from torchmetrics.text import CharErrorRate, EditDistance, WordErrorRate, WordInfoLost, WordInfoPreserved
from tqdm import tqdm

wer = WordErrorRate()
wil = WordInfoLost()
wip = WordInfoPreserved()
cer = CharErrorRate()
ed = EditDistance()


def _to_words(match: re.Match, lang="de") -> str:
    return f" {num2words(int(match.group()), lang=lang)} "


def _transform_umlauts(text: str) -> str:
    text = text.replace("ä", "ae")
    text = text.replace("ö", "oe")
    text = text.replace("ü", "ue")
    text = text.replace("ß", "ss")
    return text


def _normalize_text(text: str) -> List[str]:
    """
    Normalize the input text.

    Parameters:
    text (str): The input text.

    Returns:
    List[str]: A list of normalized tokens.
    """

    text = text.lower()

    # replace all words in brackets (), [] with empty string
    # this is useful for removing youtube specific information like [Music]
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    # num to words
    text = re.sub(r"\s([0-9]+)\s", _to_words, text)

    # transform umlauts
    text = _transform_umlauts(text)

    # remove special characters (to whitespace, then strip whitespace, then remove double whitespace)
    # this is because things like: "we-are" should be treated as "we are"
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = text.strip().replace("  ", " ")
    return text


def calculate_metrics(preds: List[str], targets: List[str]) -> dict:
    wer.reset()
    wil.reset()
    wip.reset()
    cer.reset()
    ed.reset()

    wer_val = wer(preds, targets)
    wil_val = wil(preds, targets)
    wip_val = wip(preds, targets)
    cer_val = cer(preds, targets)
    ed_val = ed(preds, targets)

    return {
        "wer": wer_val.item(),
        "wil": wil_val.item(),
        "wip": wip_val.item(),
        "cer": cer_val.item(),
        "ed": ed_val.item(),
    }


def calculate_single_wer(pred: str, target: str) -> float:
    wer.reset()
    return wer(pred, target).item()


def print_metrics(metrics: dict) -> None:
    print(f"WER: {metrics['wer']:.4f}")
    print(f"WIL: {metrics['wil']:.4f}")
    print(f"WIP: {metrics['wip']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")
    print(f"ED: {metrics['ed']:.4f}")


def whisper_inferece(model, batch, processor, torch_dtype):
    input_features = [b["input_features"] for b in batch]
    input_features = torch.stack(input_features).squeeze(1).to("cuda").to(torch_dtype)
    with torch.no_grad():
        start_time = time.time()
        output = model.generate(input_features, language="de")
        decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
        return decoded_outputs, time.time() - start_time


def wav2vec_inferece(model, batch, processor, torch_dtype):
    input_features = [b["input_features"][0] for b in batch]

    # pad the input features
    input_features_padded = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True).to("cuda")

    with torch.no_grad():
        start_time = time.time()
        output = model(input_features_padded).logits
        predicted_ids = torch.argmax(output, dim=-1)
        decoded_outputs = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return decoded_outputs, time.time() - start_time


def infer_batches(loader, inference_fn, model, processor, torch_dtype=torch.float32):
    results = defaultdict(list)
    infer_times = []
    for batch in tqdm(loader):
        decoded_outputs, t = inference_fn(model, batch, processor, torch_dtype)
        infer_times.append(t)
        for i, decoded_output in enumerate(decoded_outputs):
            results["decoded_output"].append(decoded_output.strip())
            results["normalized_decoded_output"].append(_normalize_text(decoded_output))
            results["gt"].append(batch[i]["transcript"].strip())
            results["normalized_gt"].append(_normalize_text(batch[i]["transcript"]))
            results["audio_path"].append(batch[i]["audio_path"])

            if len(results["normalized_gt"][-1]) <= 0:
                results["valid"].append(False)
                results["wer"].append(-1)
                continue
            results["valid"].append(True)
            results["wer"].append(
                calculate_single_wer(results["normalized_decoded_output"][-1], results["normalized_gt"][-1])
            )

    results_df = pd.DataFrame(results)
    return results_df, {"mean": np.mean(infer_times), "sum": np.sum(infer_times)}


if __name__ == "__main__":
    test_text = "This is a test text with 6111234 and some special characters like äöüß."
    print(_normalize_text(test_text))
