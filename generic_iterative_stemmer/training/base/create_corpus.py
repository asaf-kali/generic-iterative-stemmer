import logging
import re
from typing import List

from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import tokenize as gensim_tokenizer

from ...utils import measure_time

log = logging.getLogger(__name__)

HEBREW_PATTERN = re.compile(r"[\u0590-\u05FF \-']+")

HEBREW_SUFFIX_LETTERS = ("ן", "ם", "ץ", "ף", "ך")


def _is_only_hebrew(token: str) -> bool:
    return HEBREW_PATTERN.fullmatch(token) is not None


def _replace_suffix_letter(token: str) -> str:
    if not token.endswith(HEBREW_SUFFIX_LETTERS):
        return token
    return token[:-1] + chr(ord(token[-1]) + 1)


def hebrew_tokenizer(content: str, token_min_len: int, token_max_len: int, lower: bool) -> List[str]:
    tokens = gensim_tokenizer(content, token_min_len, token_max_len, lower)
    hebrew_tokens = [_replace_suffix_letter(token) for token in tokens if _is_only_hebrew(token)]
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
    from ...utils import get_path

    articles = get_path("hewiki-latest-pages-articles.xml.bz2")
    out = get_path("wiki-he-filtered2", "corpus.txt")
    generate_wiki_corpus_file(articles, out)
