from src.text_preprocessing import clean_text


def test_clean_text_collapses_whitespace_and_handles_missing() -> None:
    assert clean_text("  Wasser\n\nschaden\t in   Halle  ") == "Wasser schaden in Halle"
    assert clean_text(None) == ""
