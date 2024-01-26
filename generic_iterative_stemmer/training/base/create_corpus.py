import logging
import re
from typing import Callable, List, Set

from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import tokenize
from tqdm import tqdm

log = logging.getLogger(__name__)

HEBREW_LETTER_PATTERN = re.compile(r"[\u0590-\u05FF]")
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

TokenizerFunc = Callable[[str, int, int, bool], List[str]]


class HebrewTokenizer:
    def __init__(self, token_min_len: int, token_max_len: int, replace_hebrew_suffix: bool):
        self.token_min_len = token_min_len
        self.token_max_len = token_max_len
        self.replace_hebrew_suffix = replace_hebrew_suffix
        self.filtered_tokens: Set[str] = set()

    def tokenize(self, content: str) -> List[str]:
        content_without_scores = HEBREW_WIKI_REPLACE_PATTERN.sub("", content)
        hebrew_tokens = [match.group() for match in HEBREW_WORD_PATTERN.finditer(content_without_scores)]
        length_filtered = filter(self._should_keep_token, hebrew_tokens)
        if not self.replace_hebrew_suffix:
            return list(length_filtered)
        no_suffix = [_replace_to_non_suffix_letter(token) for token in length_filtered]
        # log.debug(f"Filtered {len(self.filtered_tokens)} tokens: {self.filtered_tokens}")
        return no_suffix

    def _should_keep_token(self, token: str) -> bool:
        token_len = len(token)
        if not self.token_min_len <= token_len <= self.token_max_len:
            return False
        if len(HEBREW_LETTER_PATTERN.findall(token)) < token_len <= 3:
            self.filtered_tokens.add(token)
            return False
        return True


def _replace_to_non_suffix_letter(token: str) -> str:
    if not token.endswith(HEBREW_SUFFIX_LETTERS):
        return token
    suffix_letter = token[-1]
    return token[:-1] + SUFFIX_LETTER_TO_NON_SUFFIX_LETTER[suffix_letter]


def hebrew_tokenizer(
    content: str, token_min_len: int, token_max_len: int, lower: bool = False, replace_hebrew_suffix: bool = False
) -> List[str]:
    tokenizer = HebrewTokenizer(
        token_min_len=token_min_len, token_max_len=token_max_len, replace_hebrew_suffix=replace_hebrew_suffix
    )
    return tokenizer.tokenize(content)


def hebrew_tokenizer_no_suffix(content: str, token_min_len: int, token_max_len: int, lower: bool = False) -> List[str]:
    return hebrew_tokenizer(
        content=content,
        token_min_len=token_min_len,
        token_max_len=token_max_len,
        lower=lower,
        replace_hebrew_suffix=True,
    )


def generate_wiki_corpus_file(articles_file_path: str, output_file_path: str, tokenizer_func: TokenizerFunc = tokenize):
    log.info("Creating wiki corpus")
    article_count = 0
    with open(output_file_path, "w") as corpus_file:
        wiki = WikiCorpus(fname=articles_file_path, dictionary={}, tokenizer_func=tokenizer_func)
        for article_words in tqdm(wiki.get_texts(), desc="Generate wiki corpus", smoothing=0.1):
            article = " ".join(article_words)
            corpus_file.write(f"{article}\n")
            article_count += 1
    log.info(f"Finished - Saved {article_count} articles")
