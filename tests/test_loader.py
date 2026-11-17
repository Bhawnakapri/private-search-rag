from app.loader import clean_markdown


def test_table_rows_become_separate_sentences():
    md = "| Model | AUC |\n|---|---|\n| Logistic | 0.84 |\n| XGBoost | 0.83 |\n\nNext paragraph."
    out = clean_markdown(md)
    assert "|" not in out and "---" not in out
    assert "Logistic, 0.84." in out and "XGBoost, 0.83." in out
    assert "0.83.\n\nNext paragraph." in out


def test_quote_bold_and_heading_markers_removed():
    out = clean_markdown("## Results\n\n> **Bold** quote\n")
    assert out.startswith("Results.") and "**" not in out and ">" not in out