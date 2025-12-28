# tests/test_smoke.py

def test_smoke_imports():
    """
    Basic smoke test to ensure the package imports correctly.
    """
    import nlp_imdb
    import nlp_imdb.cli
    import nlp_imdb.data
    import nlp_imdb.preprocessing
    import nlp_imdb.features
    import nlp_imdb.models
    import nlp_imdb.training
    import nlp_imdb.utils

    assert True
