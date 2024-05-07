import re
from typing import List

import editdistance
from num2words import num2words

# code based on https://deepgram.com/learn/benchmarking-openai-whisper-for-non-english-asr


def _wer(truth: str, pred: str, tokenizer) -> float:
    """
    Calculate Word Error Rate (WER) between truth and prediction.

    Parameters:
    truth (str): The ground truth text.
    pred (str): The predicted text.
    tokenizer (function): A function to tokenize the input strings.

    Returns:
    float: The Word Error Rate between truth and prediction.
    """
    truth_tokens = tokenizer(truth)
    pred_tokens = tokenizer(pred)
    return editdistance.eval(truth_tokens, pred_tokens) / len(truth_tokens)


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

    # num to words
    text = re.sub(r"\s([0-9]+)\s", _to_words, text)
    # transform umlauts
    text = _transform_umlauts(text)
    # remove special characters (to whitespace, then strip whitespace, then remove double whitespace)
    # this is because things like: "we-are" should be treated as "we are"
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = text.strip().replace("  ", " ")
    return text.split()


if __name__ == "__main__":
    truth = "[Musik] Die vier ist in der Küche / und die 5 ist im Garten / und die 6 ist im Wohnzimmer"
    pred = "die 4 ist in der küche und die 5 ist im Garten und die 6 ist im Wohnzimmer"

    print(f"Truth: {truth}")
    print(f"Normalized truth: {_normalize_text(truth)}")
    print(f"Prediction: {pred}")
    print(f"Normalized prediction: {_normalize_text(pred)}")
    print(f"WER: {_wer(truth, pred, _normalize_text)}")
