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

This section shows the basic flow this repository was designed to perform. It
supports more complicated flows as well.

The output of the training process is a `StemmedKeyedVectors` object (in the
form of a `.kv` file), which inherits the standard `gensim.models.KeyedVectors`.

1. Under `./data` folder, create a directory for your corpus (for
   example, `wiki-he`).
2. Download Hebrew (or any other language) dataset from Wikipedia:
    1. Go to [wikimedia dumps](https://dumps.wikimedia.org/hewiki/latest/).
    2. Download `hewiki-latest-pages-articles.xml.bz2`, and save it
       under `./data/wiki-he`.
3. Create your initial text corpus:
   TODO: create a notebook for that.
4. Train the model:
   TODO: create a notebook for that.
5. Play with your trained model using `playground.ipynb`.

## Generic iterative stemming

TODO: Explain the algorithm.
