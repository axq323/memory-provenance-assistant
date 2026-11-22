from mpa.display import tame_headings


def test_headings_become_bold():
    text = "# Request for Friday Afternoon Off\n\nDear Sam,\n\n## Details ##\nText"
    assert tame_headings(text) == "**Request for Friday Afternoon Off**\n\nDear Sam,\n\n**Details**\nText"


def test_other_hashes_are_left_alone():
    assert tame_headings("Use #hashtags and C# code") == "Use #hashtags and C# code"
