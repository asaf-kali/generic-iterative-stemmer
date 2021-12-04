import logging

import numpy as np
from gensim.matutils import unitvec

import fasttxt
import word2vec

log = logging.getLogger(__name__)


def test(model, positive, negative, test_words):
    mean = []
    for pos_word in positive:
        mean.append(1.0 * np.array(model[pos_word]))

    for neg_word in negative:
        mean.append(-1.0 * np.array(model[neg_word]))

    # compute the weighted average of all words
    mean = unitvec(np.array(mean).mean(axis=0))

    scores = {}
    for word in test_words:

        if word not in positive + negative:
            test_word = unitvec(np.array(model[word]))

            # Cosine Similarity
            scores[word] = np.dot(test_word, mean)

    log.info(sorted(scores, key=scores.get, reverse=True)[:1])


TRAIN = False

if TRAIN:
    # log.info("Training Word2vec")
    # word2vec.train()

    log.info("Training Fasttext")
    fasttxt.train()

positive_words = ["מלכה", "גבר"]

negative_words = ["מלך"]

# Test Word2vec
log.info("Testing Word2vec")
model = word2vec.getModel()
test(model, positive_words, negative_words, model.vocab)

# # Test Fasttext
# log.info("Testing Fasttext")
# model = fasttxt.getModel()
# test(model, positive_words, negative_words, model.words)
