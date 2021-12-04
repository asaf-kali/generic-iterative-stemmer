import logging
import time

import fasttext

log = logging.getLogger(__name__)


def train(inp="wiki-he.txt", out_model="wiki-he.fasttext.model", alg="CBOW"):
    start = time.time()
    if alg == "skipgram":
        # Skipgram model
        model = fasttext.skipgram(inp, out_model)
    else:
        # CBOW model
        model = fasttext.train_unsupervised(inp, out_model)
    log.info(model.words)
    log.info(time.time() - start)
    model.save(out_model)


def getModel(model="wiki-he.fasttext.model.bin"):
    model = fasttext.load_model(model)

    return model
