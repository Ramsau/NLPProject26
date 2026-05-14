import csv
from typing import List, Any, Dict, Tuple, Callable, Optional
import re

from ngrams import build_knowledge_base as build_ngram_knowledge_base
from ngrams import classify as classify_ngrams
from ngrams import preprocess_text as preprocess_text_ngram
from pos import build_knowledge_base as build_pos_knowledge_base
from pos import preprocess_text as preprocess_text_pos

type KnowledgeBase = Any
type TextSet = List[str]
type ClassifyFunction = Callable[[List[List[str]]], float]
type TestResult = List[Tuple[str, str, float]]



def test(data: List[List[str]], preprocess: Callable[[str], str], classify: Callable[[str, KnowledgeBase, Optional[int]], float], knowledge_base: KnowledgeBase, verbose=False) -> TestResult:
    classified_data = []
    for entry in data:
        preprocessed_text = preprocess(entry[0])
        rating = classify(preprocessed_text, knowledge_base)
        if rating > 0 != (entry[1].lower() == 'opinion') and verbose:
            print(f"Misclassification as {'Fact' if rating > 0 else 'Opinion'}({rating:.6f}), True: {entry[1]}")
            print(entry[0])
            print(preprocessed_text)
            classify(preprocessed_text, knowledge_base, 10)
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

    train_facts = [entry[0] for entry in train_data if entry[1].lower() == 'fact']
    train_opinions = [entry[0] for entry in train_data if entry[1].lower() == 'opinion']

    normalize_ngram_average = True
    normalize_ngram_occurrences = True
    laplace_smoothing = 4
    ignore_ngrams = []

    train_facts_pp_ngram = [preprocess_text_ngram(text) for text in train_facts]
    train_opinions_pp_ngram = [preprocess_text_ngram(text) for text in train_opinions]
    ngram_factuality_ratings = build_ngram_knowledge_base(train_facts_pp_ngram, train_opinions_pp_ngram,
                                                          normalize_ngram_average,
                                                          normalize_ngram_occurrences,
                                                          laplace_smoothing,
                                                          ignore_ngrams)


    trial = test(test_data, preprocess_text_ngram, classify_ngrams, ngram_factuality_ratings, verbose=True)

    misclassified = [entry for entry in trial if (entry[2] < 0) != (entry[1].lower() == 'opinion')]
    print(f"Test facts: {len([entry for entry in test_data if entry[1].lower() == 'fact'])}, Test opinions: {len([entry for entry in test_data if entry[1].lower() == 'opinion'])}")
    print(f"Misclassification rate: {len(misclassified)}/{len(trial)} ({len(misclassified) / len(trial) * 100:.2f}%)")

    train_facts_pp_pos = [preprocess_text_pos(text) for text in train_facts]
    train_opinions_pp_pos = [preprocess_text_pos(text) for text in train_opinions]
    pos_factuality_ratings = build_pos_knowledge_base(train_facts_pp_pos, train_opinions_pp_pos)

    trial = test(test_data, preprocess_text_pos, classify_ngrams, pos_factuality_ratings, verbose=True)
    misclassified = [entry for entry in trial if (entry[2] < 0) != (entry[1].lower() == 'opinion')]
    print(f"Misclassification rate (POS): {len(misclassified)}/{len(trial)} ({len(misclassified) / len(trial) * 100:.2f}%)")




if __name__ == '__main__':
    main()
