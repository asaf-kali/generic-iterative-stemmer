import logging
import re
from typing import Callable, List

from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import tokenize
from tqdm import tqdm

log = logging.getLogger(__name__)

HEBREW_WORD_PATTERN = re.compile(r"[\u0590-\u05FF\-']+")
HEBREW_WIKI_REPLACE_PATTERN = re.compile(r"[\u0591-\u05C7]|'''|''")
JUNK_PATTERN = re.compile(r"''")
SUFFIX_LETTER_TO_NON_SUFFIX_LETTER = {
    "ך": "כ",
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
}
HEBREW_SUFFIX_LETTERS = tuple(SUFFIX_LETTER_TO_NON_SUFFIX_LETTER.keys())


def _replace_to_non_suffix_letter(token: str) -> str:
    if not token.endswith(HEBREW_SUFFIX_LETTERS):
        return token
    suffix_letter = token[-1]
    return token[:-1] + SUFFIX_LETTER_TO_NON_SUFFIX_LETTER[suffix_letter]


def hebrew_tokenizer(content: str, token_min_len: int, token_max_len: int, lower: bool = False) -> List[str]:
    content_without_scores = HEBREW_WIKI_REPLACE_PATTERN.sub("", content)
    hebrew_tokens = [match.group() for match in HEBREW_WORD_PATTERN.finditer(content_without_scores)]
    length_filtered = filter(lambda token: token_min_len <= len(token) <= token_max_len, hebrew_tokens)
    no_suffix = [_replace_to_non_suffix_letter(token) for token in length_filtered]
    return no_suffix


def generate_wiki_corpus_file(articles_file_path: str, output_file_path: str, tokenizer_func: Callable = tokenize):
    log.info("Creating wiki corpus")
    article_count = 0
    with open(output_file_path, "w") as corpus_file:
        wiki = WikiCorpus(fname=articles_file_path, dictionary={}, tokenizer_func=tokenizer_func)
        for article_words in tqdm(wiki.get_texts(), desc="Generate wiki corpus", smoothing=0.1):
            article = " ".join(article_words)
            corpus_file.write(f"{article}\n")
            article_count += 1
    log.info(f"Finished - Saved {article_count} articles")


if __name__ == "__main__":
    from ...utils import configure_logging, get_path

    configure_logging()
    articles = get_path("hewiki-latest-pages-articles.xml.bz2")
    out = get_path("wiki-he", "corpus.txt")
    generate_wiki_corpus_file(articles, out, tokenizer_func=hebrew_tokenizer)
