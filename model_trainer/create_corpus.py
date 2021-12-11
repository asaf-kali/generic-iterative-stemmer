import logging
import re
from typing import List

from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import tokenize as gensim_tokenizer

from model_trainer.config import get_path
from utils.logging import measure_time

log = logging.getLogger(__name__)

HEBREW_PATTERN = re.compile(r"[\u0590-\u05FF \-']+")


def _is_only_hebrew(token: str) -> bool:
    return HEBREW_PATTERN.fullmatch(token) is not None


def hebrew_tokenizer(content: str, token_min_len: int, token_max_len: int, lower: bool) -> List[str]:
    tokens = gensim_tokenizer(content, token_min_len, token_max_len, lower)
    hebrew_tokens = [token for token in tokens if _is_only_hebrew(token)]
    return hebrew_tokens


@measure_time
def generate_wiki_corpus_file(articles_file_path: str, output_file_path: str):
    log.info("Creating wiki corpus")
    article_count = 0
    with open(output_file_path, "w") as corpus_file:
        wiki = WikiCorpus(fname=articles_file_path, dictionary={}, tokenizer_func=hebrew_tokenizer)
        for article_words in wiki.get_texts():
            article = " ".join(article_words)
            corpus_file.write(f"{article}\n")
            article_count += 1
            if article_count % 1000 == 0:
                log.debug(f"Saved {article_count} articles")
    log.info(f"Finished - Saved {article_count} articles")


if __name__ == "__main__":
    articles = get_path("hewiki-latest-pages-articles.xml.bz2")
    out = get_path("wiki-he-filter", "corpus2.txt")
    generate_wiki_corpus_file(articles, out)
