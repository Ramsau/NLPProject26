import csv
from typing import List, Any, Dict, Tuple
import re

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

def process_ngrams(entries: List[str], laplace_smoothing: int) -> Dict[int, Dict[str, Tuple[int, float, float]]]:
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

def get_ngram_differences(ngrams_a: Dict[int, Dict[str, Tuple[int, float, float]]], ngrams_b: Dict[int, Dict[str, Tuple[int, float, float]]], normalize_occurrences=False) -> Dict[int, Dict[str, float]]:
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

def normalize_averages(differences: Dict[int, Dict[str, float]]) -> None:
    for n in range(1, 5):
        average = sum(differences[n].values()) / len(differences[n])
        for ngram in differences[n].keys():
            differences[n][ngram] -= average

def remove_ngrams(ngrams: Dict[int, Dict[str, float]], ignore_ngrams: List[str]) -> None:
    for ngram in ignore_ngrams:
        for n in range(1, 5):
            if ngram in ngrams[n]:
                del ngrams[n][ngram]

def classify(text: str, ngram_differences: Dict[int, Dict[str, float]], print_ratings = 0) -> float:
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
        print(f"{text}:")
        for ngram, rating in ratings:
            print(f"{ngram}: {rating:.7f}")
    return estimate

def test(data: List[List[str]], ngram_differences: Dict[int, Dict[str, float]]) -> List[Tuple[str, str, float]]:
    classified_data = []
    for entry in data:
        rating = classify(entry[0], ngram_differences)
        if rating > 0 != (entry[1].lower() == 'opinion'):
            print(f"Misclassification as {'Fact' if rating > 0 else 'Opinion'}({rating:.6f}), True: {entry[1]}")
            classify(entry[0], ngram_differences, 10)
            print()
        classified_data.append((entry[0], entry[1], rating))
    return classified_data

def main() -> None:
    with open('../dataset.csv', 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # discard header
    data.pop(0)
    train_idx = int(len(data) * 0.8)
    train_data = data[:train_idx]
    test_data = data[train_idx:]

    train_facts = [preprocess_text(entry[0]) for entry in train_data if entry[1].lower() == 'fact']
    train_opinions = [preprocess_text(entry[0]) for entry in train_data if entry[1].lower() == 'opinion']

    normalize_ngram_average = True
    normalize_ngram_occurrences = True
    laplace_smoothing = 4
    ignore_ngrams = []

    train_fact_ngrams = process_ngrams(train_facts, laplace_smoothing)
    train_opinion_ngrams = process_ngrams(train_opinions, laplace_smoothing)

    ngram_factuality_ratings = get_ngram_differences(train_fact_ngrams, train_opinion_ngrams, normalize_ngram_occurrences)
    if normalize_ngram_average:
        normalize_averages(ngram_factuality_ratings)
    remove_ngrams(ngram_factuality_ratings, ignore_ngrams)

    trial = test(test_data, ngram_factuality_ratings)

    misclassified = [entry for entry in trial if (entry[2] < 0) != (entry[1].lower() == 'opinion')]
    print(f"Test facts: {len([entry for entry in test_data if entry[1].lower() == 'fact'])}, Test opinions: {len([entry for entry in test_data if entry[1].lower() == 'opinion'])}")
    print(f"Misclassification rate: {len(misclassified)}/{len(trial)} ({len(misclassified) / len(trial) * 100:.2f}%)")




if __name__ == '__main__':
    main()
