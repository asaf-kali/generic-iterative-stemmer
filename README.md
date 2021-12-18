# Word embedding: generic iterative stemmer

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

[![Lint](https://github.com/asaf-kali/gensim-model-trainer/actions/workflows/lint.yml/badge.svg)](https://github.com/asaf-kali/gensim-model-trainer/actions/workflows/lint.yml)
[![Tests](https://github.com/asaf-kali/gensim-model-trainer/actions/workflows/tests.yml/badge.svg)](https://github.com/asaf-kali/gensim-model-trainer/actions/workflows/tests.yml)

## Setup

1. Create a `python3` virtual environment.
2. Install dependencies using `make install`.

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
