# Word embedding: generic iterative stemmer

[![PyPI version](https://badge.fury.io/py/generic-iterative-stemmer.svg)](https://badge.fury.io/py/generic-iterative-stemmer)
[![Pipeline](https://github.com/asaf-kali/generic-iterative-stemmer/actions/workflows/pipeline.yml/badge.svg)](https://github.com/asaf-kali/generic-iterative-stemmer/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/github/asaf-kali/generic-iterative-stemmer/graph/badge.svg?token=HET5E8P1UK)](https://codecov.io/github/asaf-kali/generic-iterative-stemmer)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-111111.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://img.shields.io/badge/type%20check-mypy-22aa11)](http://mypy-lang.org/)
[![Linting: pylint](https://img.shields.io/badge/linting-pylint-22aa11)](https://github.com/pylint-dev/pylint)

A generic helper for training `gensim` and `fasttext` word embedding models.<br>
Specifically, this repository was created in order to
implement [stemming](https://en.wikipedia.org/wiki/Stemming)
on a Wikipedia-based corpus in Hebrew, but it will probably also work for other
corpus sources and languages as well.

**Important** to note that while there are sophisticated and efficient
approaches to the stemming task, this repository implements a naive approach
with no strict time or memory considerations (more about that in
the [explanation section](#generic-iterative-stemming)).

Based on https://github.com/liorshk/wordembedding-hebrew.

## Setup

1. Create a `python3` virtual environment.
2. Install dependencies using `make install` (this will run tests too).

## Usage

The general flow is as follows:

1. Get a text corpus (for example, from Wikipedia).
2. Create a training program.
3. Run a `StemmingTrainer`.

The output of the training process is a `generic_iterative_stemmer.models.StemmedKeyedVectors` object
(in the form of a `.kv` file). It has the same interface as the standard `gensim.models.KeyedVectors`,
so the 2 can be used interchangeably.

### 0. (Optional) Set up a language data cache

`generic_iterative_stemmer` uses a language data cache to store its output and intermediate results. \
The language data directory is useful if you want to train multiple models on the same corpus, or if you want to
train a model on a corpus that you've already trained on in the past, with different parameters.

To set up the language data cache, run ```mkdir -p ~/.cache/language_data```.

**Tip**: soft-link the language data cache to your project's root directory,
e.g. `ln -s ~/.cache/language_data language_data`.

### 1. Get a text corpus

If you don't a specific corpus in mind, you can use Wikipedia. Here's how:

1. Under `~/.cache/language_data` folder, create a directory for your corpus (for example, `wiki-he`).
2. Download Hebrew (or any other language) dataset from Wikipedia:
    1. Go to [wikimedia dumps](https://dumps.wikimedia.org/hewiki/latest/) (in the URL, replace `he` with
       your [language code](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes)).
    2. Download the matching `wiki-latest-pages-articles.xml.bz2` file, and place it in your corpus directory.
3. Create initial text corpus: run the script inside `notebooks/create_corpus.py` (change parameters as needed). \
   This will create a `corpus.txt` file in your corpus directory. It takes roughly 15 minutes to run (depending on the
   corpus size and your computer).

### 2. Create a training program

TODO

### 3. Run a `StemmingTrainer`

TODO

### 4. Play with your trained model

Play with your trained model using `playground.ipynb`.

## Generic iterative stemming

TODO: Explain the algorithm.
