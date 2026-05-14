from typing import List, Dict, Tuple, Any
import re

type CountedNgrams = Dict[int, Dict[str, Tuple[int, float, float]]]
type NgramRatings = Dict[int, Dict[str, float]]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\$?\+?-?[0-9]+((,|\.)[0-9]+)*%?', '[NUM]', text)
    text = text.replace('"', ' [QUOT] ').replace("“", ' [QUOT] ').replace("”", ' [QUOT] ')
    for symbol in ['\n', ',', '.', ':', ';', "'", '(', ')']:
        text = text.replace(symbol, ' ')
    return text

def generate_ngrams(text: str, n: int) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def count_occurrences(list: List[Any], laplace_smoothing: int) -> Dict[Any, Tuple[int, float, float]]:
    list_count = {}
    for item in list:
        if item in list_count:
            list_count[item] += 1
        else:
            list_count[item] = 1

    total_entries = len(list)
    total_different_items = len(list_count)
    list_count = {k: (v, v / total_entries, (v + laplace_smoothing) / (total_entries + laplace_smoothing * total_different_items)) for k, v in list_count.items()}

    # return sorted(list_count.items(), key=lambda x: x[1], reverse=True)
    return list_count

def process_ngrams(entries: List[str], laplace_smoothing: int) -> CountedNgrams:
    ngrams = {
        1: [],
        2: [],
        3: [],
        4: [],
    }
    for entry in entries:
        for n in range(1, 5):
            ngram_words = generate_ngrams(entry, n)
            ngrams[n].extend(ngram_words)

    ngram_occurrences = {
        1: count_occurrences(ngrams[1], laplace_smoothing),
        2: count_occurrences(ngrams[2], laplace_smoothing),
        3: count_occurrences(ngrams[3], laplace_smoothing),
        4: count_occurrences(ngrams[4], laplace_smoothing),
    }
    return ngram_occurrences

def get_ngram_differences(ngrams_a: CountedNgrams, ngrams_b: CountedNgrams, normalize_occurrences=False) -> NgramRatings:
    differences = {
        1: {},
        2: {},
        3: {},
        4: {},
    }
    for n in range(1, 5):
        for ngram in ngrams_a[n].keys():
            if ngram in ngrams_b[n]:
                relative_a = ngrams_a[n][ngram][2]
                relative_b = ngrams_b[n][ngram][2]
                differences[n][ngram] = relative_a - relative_b

                if normalize_occurrences:
                    differences[n][ngram] /= (ngrams_a[n][ngram][1] + ngrams_b[n][ngram][1])
    return differences

def normalize_averages(differences: NgramRatings) -> None:
    for n in range(1, 5):
        average = sum(differences[n].values()) / len(differences[n])
        for ngram in differences[n].keys():
            differences[n][ngram] -= average

def remove_ngrams(ngrams: NgramRatings, ignore_ngrams: List[str]) -> None:
    for ngram in ignore_ngrams:
        for n in range(1, 5):
            if ngram in ngrams[n]:
                del ngrams[n][ngram]

def classify(text: str, ngram_differences: NgramRatings, print_ratings = 0) -> float:
    estimate = 0
    ratings = []
    for n in range(1, 5):
        for ngram in generate_ngrams(text, n):
            if ngram in ngram_differences[n]:
                if print_ratings:
                    ratings.append((ngram, ngram_differences[n][ngram]))
                estimate += ngram_differences[n][ngram]

    if print_ratings:
        ratings.sort(key=lambda x: abs(x[1]), reverse=True)
        for ngram, rating in ratings:
            print(f"{ngram}: {rating:.7f}")
    return estimate


def build_knowledge_base(train_a: List[str], train_b: List[str],
                         normalize_ngram_average=False,
                         normalize_ngram_occurrences=False,
                         laplace_smoothing=0,
                         ignore_ngrams: List[str]=[]) -> NgramRatings:
    train_a_ngrams = process_ngrams(train_a, laplace_smoothing)
    train_b_ngrams = process_ngrams(train_b, laplace_smoothing)

    differences = get_ngram_differences(train_a_ngrams, train_b_ngrams, normalize_ngram_occurrences)
    if normalize_ngram_average:
        normalize_averages(differences)
    remove_ngrams(differences, ignore_ngrams)
    return differences


