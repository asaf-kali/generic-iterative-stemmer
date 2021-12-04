import os

from gensim.corpora import WikiCorpus

WIKI_ARTICLES_FILE_NAME = os.getenv("WIKI_ARTICLES_FILE_NAME", "hewiki-latest-pages-articles.xml.bz2")
OUTPUT_CORPUS_FILE_NAME = os.getenv("OUTPUT_CORPUS_FILE_NAME", "wiki-he.txt")


def generate_wiki_corpus_file(articles_file_path: str, output_file_path: str):
    print("Creating wiki corpus")
    article_count = 0
    with open(output_file_path, "w") as corpus_file:
        # lemmatize=False,
        wiki = WikiCorpus(fname=articles_file_path, dictionary={})
        for article_words in wiki.get_texts():
            article = " ".join(article_words)
            article_encoded = article.encode("utf-8")
            corpus_file.write(f"{article_encoded}\n")
            article_count += 1
            if article_count % 1000 == 0:
                print(f"Saved {article_count} articles")
    print(f"Finished - Saved {article_count} articles")


if __name__ == '__main__':
    generate_wiki_corpus_file(WIKI_ARTICLES_FILE_NAME, OUTPUT_CORPUS_FILE_NAME)
