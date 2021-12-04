import fasttext
import time


def train(inp="wiki-he.txt", out_model="wiki-he.fasttext.model", alg="CBOW"):
    start = time.time()
    if alg == "skipgram":
        # Skipgram model
        model = fasttext.skipgram(inp, out_model)
    else:
        # CBOW model
        model = fasttext.train_unsupervised(inp, out_model)
    print(model.words)
    print(time.time() - start)
    model.save(out_model)


def getModel(model="wiki-he.fasttext.model.bin"):

    model = fasttext.load_model(model)

    return model
