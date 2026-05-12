import csv
from typing import List, Any, Dict, Tuple
import re

def count_occurrences(list: List[Any]) -> Dict[Any, Tuple[int, float]]:
    list_count = {}
    for item in list:
        if item in list_count:
            list_count[item] += 1
        else:
            list_count[item] = 1

    total_entries = len(list)
    list_count = {k: (v, v / total_entries) for k, v in list_count.items()}

    # return sorted(list_count.items(), key=lambda x: x[1], reverse=True)
    return list_count

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\$?\+?-?[0-9]+((,|\.)[0-9]+)*%?', '[NUM]', text)
    text = text.replace('"', ' [QUOT] ').replace("“", ' [QUOT] ').replace("”", ' [QUOT] ')
    for symbol in ['\n', ',', '.', ':', ';', "'", '(', ')']:
        text = text.replace(symbol, ' ')
    return text

def process_ngrams(entries: List[str]) -> Dict[int, Dict[str, Tuple[int, float]]]:
    ngrams = {
        1: [],
        2: [],
        3: [],
        4: [],
    }
    for entry in entries:
        words = entry.split()
        for i in range(1, 5):
            ngram_words = [' '.join(words[j:j+i]) for j in range(len(words) - i + 1)]
            ngrams[i].extend(ngram_words)

    ngram_occurrences = {
        1: count_occurrences(ngrams[1]),
        2: count_occurrences(ngrams[2]),
        3: count_occurrences(ngrams[3]),
        4: count_occurrences(ngrams[4]),
    }
    return ngram_occurrences

def get_ngram_differences(ngrams_a: Dict[int, Dict[str, Tuple[int, float]]], ngrams_b: Dict[int, Dict[str, Tuple[int, float]]], laplace_smoothing: int) -> Dict[int, Dict[str, float]]:
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
                relative_a = ngrams_a[n][ngram][1]
                relative_b = ngrams_b[n][ngram][1]
                differences[n][ngram] = relative_a - relative_b
    return differences

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

    train_fact_ngrams = process_ngrams(train_facts)
    train_opinion_ngrams = process_ngrams(train_opinions)

    differences = get_ngram_differences(train_fact_ngrams, train_opinion_ngrams, 2)
    pass



if __name__ == '__main__':
    main()
