from typing import List, Any, Dict

import spacy

from ngrams import build_knowledge_base as build_ngram_knowledge_base
from ngrams import NgramRatings

spacy_model = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    doc = spacy_model(text)
    return ' '.join([token.pos_ for token in doc])


def build_knowledge_base(train_a: List[str], train_b: List[str],
                         normalize_ngram_bias=False,
                         normalize_ngram_gain=True,
                         normalize_ngram_occurrences=True,
                         laplace_smoothing=10,
                         ignore_ngrams=[]) -> NgramRatings:
    return build_ngram_knowledge_base(train_a, train_b,
                                      normalize_ngram_bias=False,
                                      normalize_ngram_gain=True,
                                      normalize_ngram_occurrences=True,
                                      laplace_smoothing=10,
                                      ignore_ngrams=[])
