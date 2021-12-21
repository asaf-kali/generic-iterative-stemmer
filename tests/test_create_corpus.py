from generic_iterative_stemmer.training.base.create_corpus import hebrew_tokenizer


def test_hebrew_tokenizer():
    content = "מישהו לקח לי את הפנינים, אלוף תן לי מיץ strawberry באמא'שך"
    expected_tokens = ["מישהו", "לקח", "לי", "את", "הפנינימ", "אלופ", "תנ", "לי", "מיצ", "באמא", "שכ"]
    tokenized_content = hebrew_tokenizer(content, token_min_len=2, token_max_len=20)
    assert tokenized_content == expected_tokens
