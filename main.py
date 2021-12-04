import logging

import numpy as np
from gensim.matutils import unitvec

from model_trainer import fast_text
from model_trainer import word2vec
from model_trainer.config import get_path

log = logging.getLogger(__name__)

positive_words = ["מלכה", "גבר"]

negative_words = ["מלך"]

# Test Word2vec
log.info("Testing Word2vec")

kv_path = get_path("wiki-he", "wiki-he.kv")
model = word2vec.load_vectors_from_kv(kv_path)

# path = get_data("wiki-he", "sg.model")
# model = word2vec.load_vectors_from_model(path)
model.most_similar(positive_words, negative_words)

# Test Fasttext
log.info("Testing Fasttext")
model = fast_text.get_model()
test(model, positive_words, negative_words, model.words)
