import pytest

from mpa.task_types import classify, label


@pytest.mark.parametrize("text, expected", [
    # the three demo prompts
    ("Can you suggest something I could make for dinner tonight?", "cooking"),
    ("I don't really understand how solar panels work. Can you explain it properly, step by step?", "explain"),
    ("Can you write a birthday message for my best friend Sam? Something warm and a bit funny.", "personal_message"),
    # other task types
    ("Write an email to my manager asking for Friday off.", "work_writing"),
    ("Can you give me feedback on my essay intro?", "feedback"),
    ("Summarise the plot of Hamlet in a few lines.", "summarise"),
    ("Plan a weekend trip to Washington DC", "plan"),
    ("Hello there", "other"),
    # personal messages that used to fall through to 'other' (live testing, E11/E12)
    ("Write a quick text to my flatmate asking them to pick up milk on the way home.", "personal_message"),
    ("Write a message for my friend Priya, whose grandfather just passed away.", "personal_message"),
    ("Write a thank-you text to my grandmother for the birthday gift she sent.", "personal_message"),
    ("Send a text message to my sister about Sunday", "personal_message"),
    # ...without pulling work messages in
    ("Write a message to my manager about the deadline.", "work_writing"),
    ("Write an email to my manager asking for Friday afternoon off.", "work_writing"),
    ("Write the opening paragraph of a cover letter for a data analyst job.", "work_writing"),
])
def test_prompts_classify(text, expected):
    assert classify(text)[0] == expected


def test_trigger_phrase_is_reported():
    assert classify("Write a birthday card for Sam") == ("personal_message", "birthday")


def test_personal_message_wins_over_cooking():
    # "dinner" appears, but the request is a message to a friend
    assert classify("Write a birthday message inviting my best friend to dinner")[0] == "personal_message"


def test_word_boundaries_avoid_false_matches():
    # "planet" must not trigger "plan"; "cookie" must not trigger "cook"
    assert classify("Tell me a fact about a planet")[0] == "other"
    assert classify("Clear my cookies")[0] == "other"


def test_labels_are_readable():
    assert label("personal_message") == "personal messages"
