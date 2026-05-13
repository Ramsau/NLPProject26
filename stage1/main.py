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

def get_ngram_differences(ngrams_a: Dict[int, Dict[str, Tuple[int, float, float]]], ngrams_b: Dict[int, Dict[str, Tuple[int, float, float]]]) -> Dict[int, Dict[str, float]]:
    differences = {
        1: {},
        2: {},
        3: {},
        4: {},
    }
    for n in range(1, 5):
        for ngram in ngrams_a[n].keys():
            if ngram in ngrams_b[n]:
                # TODO implement laplace smoothing
                relative_a = ngrams_a[n][ngram][2]
                relative_b = ngrams_b[n][ngram][2]
                differences[n][ngram] = relative_a - relative_b
    return differences

def classify(text: str, ngram_differences: Dict[int, Dict[str, float]]) -> float:
    estimate = 0
    for n in range(1, 5):
        for ngram in generate_ngrams(text, n):
            if ngram in ngram_differences[n]:
                estimate += ngram_differences[n][ngram]
    return estimate

def test(data: List[List[str]], ngram_differences: Dict[int, Dict[str, float]]) -> List[Tuple[str, str, float]]:
    classified_data = []
    for entry in data:
        classified_data.append((entry[0], entry[1], classify(entry[0], ngram_differences)))
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

    train_fact_ngrams = process_ngrams(train_facts, 5)
    train_opinion_ngrams = process_ngrams(train_opinions, 5)

    differences = get_ngram_differences(train_fact_ngrams, train_opinion_ngrams)

    trial = test(test_data, differences)

    misclassified = [entry for entry in trial if (entry[2] < 0) != (entry[1].lower() == 'opinion')]
    # TODO normalize differences
    pass



if __name__ == '__main__':
    main()
