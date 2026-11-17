import pytest

from app.engine import SearchEngine


def test_indexes_sample_documents(engine):
    assert len(engine) >= 6


@pytest.mark.parametrize("question,source", [
    ("how many days of annual leave", "leave_policy.md"),
    ("vpn will not connect", "vpn_setup.md"),
    ("receipt needed for expense claim", "expense_policy.md"),
    ("postmortem after incident", "incident_response.md"),
])
def test_top_hit_is_the_right_document(engine, question, source):
    assert engine.search(question, k=3)[0].source == source


def test_hybrid_hits_carry_all_scores(engine):
    hit = engine.search("annual leave carry forward", k=1)[0]
    assert hit.rrf > 0 and hit.bm25 > 0 and hit.coverage > 0


def test_each_mode_works(engine):
    for mode in ("bm25", "dense", "hybrid"):
        assert engine.search("password reset link", k=2, mode=mode)


def test_invalid_mode_rejected(engine):
    with pytest.raises(ValueError):
        engine.search("x", mode="magic")


def test_save_and_load_roundtrip(engine, cfg, tmp_path):
    engine.save(str(tmp_path / "saved"))
    loaded = SearchEngine(cfg)
    loaded.load(str(tmp_path / "saved"))
    a, b = engine.search("sick leave doctor note", k=3), loaded.search("sick leave doctor note", k=3)
    assert [h.id for h in a] == [h.id for h in b]


def test_loading_with_a_different_embedder_is_refused(engine, cfg, tmp_path):
    from dataclasses import replace
    from app.embedder import HashingEmbedder

    engine.save(str(tmp_path / "saved"))
    other = SearchEngine(replace(cfg), embedder=HashingEmbedder(dim=128))
    with pytest.raises(ValueError):
        other.load(str(tmp_path / "saved"))


def test_missing_folder_raises(cfg):
    with pytest.raises(FileNotFoundError):
        SearchEngine(cfg).index_folder("does/not/exist")


def test_pdf_pages_are_indexed(cfg, tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "docs" / "manual.pdf"
    pdf.parent.mkdir()
    c = canvas.Canvas(str(pdf), pagesize=A4)
    c.drawString(72, 750, "Warranty covers manufacturing defects for twenty four months.")
    c.showPage()
    c.drawString(72, 750, "Battery replacement is free during the first year.")
    c.save()
    e = SearchEngine(cfg)
    e.index_folder(str(pdf.parent))
    hit = e.search("battery replacement", k=1)[0]
    assert (hit.source, hit.page) == ("manual.pdf", 2)
