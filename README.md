# Stemming word embedding trainer

A generic helper for training `gensim` and `fasttext` word embedding models.<br>
Specifically, this repository was created to implement word stemming on a Wikipedia-based corpus in Hebrew, but it will
probably also work for other languages and corpus sources as well.

**Important** to note that while there are efficient and sophisticated approaches to the stemming task, this repository
implements a naive approach, with no real-life time or memory considerations.

Based on https://github.com/liorshk/wordembedding-hebrew.

## Setup

1. Create a `python3` virtual environment.
2. Install dependencies using `make install`.

## Usage

This section shows the basic flow this repository was designed to perform.<br>
It supports more complicated flows as well.

1. Under `./data` folder, create a directory for your corpus (for example, `wiki-he`).


2. Download Hebrew (or any other language) dataset from Wikipedia:
    1. Go to [wikimedia dumps](https://dumps.wikimedia.org/hewiki/latest/).
    2. Download `hewiki-latest-pages-articles.xml.bz2`, and save it under `./data/wiki-he`.


3. Create your initial text corpus:

   TODO: create a notebook for that.


4. Train the model:

   TODO: create a notebook for that.


5. Play with your trained model using `playground.ipynb`.
