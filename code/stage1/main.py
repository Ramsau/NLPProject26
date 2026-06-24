import csv
from doctest import TestResults
from typing import List, Any, Dict, Tuple, Callable, Optional
import re

from ngrams import build_knowledge_base as build_ngram_knowledge_base
from ngrams import classify as classify_ngrams
from ngrams import preprocess_text as preprocess_text_ngram
from pos import build_knowledge_base as build_pos_knowledge_base
from pos import preprocess_text as preprocess_text_pos

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

type KnowledgeBase = Any
type TextSet = List[str]
type ClassifyFunction = Callable[[List[List[str]]], float]
type TestResult = List[Tuple[str, str, float, List]]



def run_test(data: List[List[str]], preprocess: Callable[[str], str], classify: Callable[[str, KnowledgeBase, Optional[int]], Tuple[float, List]], knowledge_base: KnowledgeBase, verbose=False, explanation_depth=0) -> TestResult:
    classified_data = []
    for entry in data:
        preprocessed_text = preprocess(entry[0])
        rating, explanations = classify(preprocessed_text, knowledge_base, print_ratings=explanation_depth)
        if verbose:
            if (rating > 0) != (entry[1].lower() == 'fact'):
                class_string = "Misclassification"
            else:
                class_string = "Classification"
            print(f"{class_string} as {'Fact' if rating > 0 else 'Opinion'}({rating:.6f}), True: {entry[1]}")
            print(entry[0])
            print(preprocessed_text)
            # classify(preprocessed_text, knowledge_base, 10)
            print()
        if explanation_depth:
            for ngram, rating in explanations[:explanation_depth]:
                print(f"{ngram}: {rating:.7f}")

        classified_data.append((entry[0], entry[1], rating, explanations))
    return classified_data

def combine_classifications(ngram_results: TestResult, pos_results: TestResult, verbose = False, explanation_depth = 0) -> TestResult:
    results = []
    for ngram_result, pos_result in zip(ngram_results, pos_results):
        classification = (ngram_result[2] + pos_result[2]) / 2
        results.append((ngram_result[0], ngram_result[1], classification))

        if verbose:
            if (classification > 0) != (ngram_result[1].lower() == 'fact'):
                class_string = "Misclassification"
            else:
                class_string = "Classification"
            print(f"{class_string} as {'Fact' if classification > 0 else 'Opinion'}({classification:.6f}, {ngram_result[2]:.6f}/{pos_result[2]:.6f}), True: {ngram_result[1]}")
            print(ngram_result[0])
            print()

        if explanation_depth > 0:
            explanations = ngram_result[3] + pos_result[3]
            explanations.sort(key=lambda x: abs(x[1]), reverse=True)
            explanations = [(ngram, rating / 2) for ngram, rating in explanations]
            for explanation in explanations[:explanation_depth]:
                print(f"{explanation[0]}: {explanation[1]:.7f}")

    return results

def build_knowledge_base(data):
    data_fact = [entry[0] for entry in data if entry[1].lower() == 'fact']
    data_opinion = [entry[0] for entry in data if entry[1].lower() == 'opinion']

    train_facts_pp_ngram = [preprocess_text_ngram(text) for text in data_fact]
    train_opinions_pp_ngram = [preprocess_text_ngram(text) for text in data_opinion]
    # ngram baseline
    baseline_factuality_ratings = build_ngram_knowledge_base(train_facts_pp_ngram, train_opinions_pp_ngram,
                                                             normalize_ngram_bias=False,
                                                             normalize_ngram_gain=False,
                                                             normalize_ngram_occurrences=False,
                                                             laplace_smoothing=0,
                                                             ignore_ngrams=[])

    # ngram
    ngram_factuality_ratings = build_ngram_knowledge_base(train_facts_pp_ngram, train_opinions_pp_ngram,
                                                          normalize_ngram_bias=False,
                                                          normalize_ngram_gain=True,
                                                          normalize_ngram_occurrences=True,
                                                          laplace_smoothing=4,
                                                          ignore_ngrams=[])

    # POS ngram
    train_facts_pp_pos = [preprocess_text_pos(text) for text in data_fact]
    train_opinions_pp_pos = [preprocess_text_pos(text) for text in data_opinion]
    pos_factuality_ratings = build_pos_knowledge_base(train_facts_pp_pos, train_opinions_pp_pos,
                                                      normalize_ngram_bias=False,
                                                      normalize_ngram_gain=True,
                                                      normalize_ngram_occurrences=True,
                                                      laplace_smoothing=4,
                                                      ignore_ngrams=[])

    return baseline_factuality_ratings, ngram_factuality_ratings, pos_factuality_ratings

def score_test(test_results):
    true_fact = len([entry for entry in test_results if (entry[2] > 0) and (entry[1].lower() == 'fact')])
    true_opinion = len([entry for entry in test_results if (entry[2] <= 0) and (entry[1].lower() == 'opinion')])
    false_fact = len([entry for entry in test_results if (entry[2] > 0) and (entry[1].lower() == 'opinion')])
    false_opinion = len([entry for entry in test_results if (entry[2] <= 0) and (entry[1].lower() == 'fact')])

    if true_fact != 0 and false_fact != 0 and false_opinion != 0:
        precision = true_fact / (true_fact + false_fact)
        recall = true_fact / (true_fact + false_opinion)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        f1 = 0

    labels_true = [entry[1].lower() for entry in test_results]
    labels_pred = ['opinion' if entry[2] <= 0 else 'fact' for entry in test_results]
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(8, 6))
    conf_matrix = ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, cmap='Blues', ax=ax)
    # plt.title(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    plt.show()

    return precision, recall, f1


def test(baseline_kb, ngram_kb, pos_kb, data, verbose=False, explanation_depth=0):
    trial_baseline = run_test(data, preprocess_text_ngram, classify_ngrams, baseline_kb, verbose=verbose, explanation_depth=explanation_depth)
    print(f"Baseline - precision, recall, f1: {score_test(trial_baseline)}")

    trial_ngram = run_test(data, preprocess_text_ngram, classify_ngrams, ngram_kb, verbose=verbose, explanation_depth=explanation_depth)
    print(f"Ngram - precision, recall, f1: {score_test(trial_ngram)}")

    trial_pos = run_test(data, preprocess_text_pos, classify_ngrams, pos_kb, verbose=verbose, explanation_depth=explanation_depth)
    print(f"POS - precision, recall, f1: {score_test(trial_pos)}")

    trial_combined = combine_classifications(trial_ngram, trial_pos, verbose=verbose, explanation_depth=explanation_depth)
    print(f"Combined - precision, recall, f1: {score_test(trial_combined)}")

def classify(ngram_kb, pos_kb, data):

    trial_ngram = run_test(data, preprocess_text_ngram, classify_ngrams, ngram_kb, verbose=False, explanation_depth=0)
    trial_pos = run_test(data, preprocess_text_pos, classify_ngrams, pos_kb, verbose=False, explanation_depth=0)
    trial_combined = combine_classifications(trial_ngram, trial_pos, verbose=False, explanation_depth=0)
    return trial_combined

def main() -> None:
    with open('../dataset.csv', 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # discard header
    data.pop(0)
    train_idx = int(len(data) * 0.8)
    train_data = data[:train_idx]
    test_data = data[train_idx:]

    print(f"Test facts: {len([entry for entry in test_data if entry[1].lower() == 'fact'])}, Test opinions: {len([entry for entry in test_data if entry[1].lower() == 'opinion'])}")
    baseline_kb, ngram_kb, pos_kb = build_knowledge_base(train_data)
    test(baseline_kb, ngram_kb, pos_kb, test_data, verbose=False, explanation_depth=0)

    print()
    print("Training and testing on all data")
    baseline_kb, ngram_kb, pos_kb = build_knowledge_base(data)
    test(baseline_kb, ngram_kb, pos_kb, data, verbose=False, explanation_depth=0)


    print()
    print("Explanation showcase")
    test(baseline_kb, ngram_kb, pos_kb, [["I like trains a lot", 'opinion']], verbose=True, explanation_depth=10)

    print()
    print("Validation dataset...")
    with open("../validationset.csv", 'r') as file:
        csv_reader = csv.reader(file)
        val_data = list(csv_reader)
    # discard header
    val_data.pop(0)
    val_data = [[entry[0], 'unknown'] for entry in val_data]

    class_data = classify(ngram_kb, pos_kb, val_data)
    classification_output = [["ID", "Verdict"]] + [[id + 1, 'Fact' if entry[2] > 0 else 'Opinion'] for id, entry in enumerate(class_data)]
    with open("../group29_classifications_1.csv", 'w+') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(classification_output)

if __name__ == '__main__':
    main()
