from generic_iterative_stemmer.training.base.create_corpus import hebrew_tokenizer


def test_hebrew_tokenizer():
    content = "מישהו לקח פנינים, י אלוף תן מיץ strawberry באמא'שך. מָתֵמָטִיקָה בבית-ספר"
    expected_tokens = [
        "מישהו",
        "לקח",
        "פנינימ",
        "אלופ",
        "תנ",
        "מיצ",
        "באמא'שכ",
        "מתמטיקה",
        "בבית-ספר",
    ]
    tokenized_content = hebrew_tokenizer(content, token_min_len=2, token_max_len=20)
    assert tokenized_content == expected_tokens
