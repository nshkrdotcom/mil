from __future__ import annotations


def test_neuronpedia_search_links_use_sae_id(monkeypatch):
    from types import SimpleNamespace

    captured = {}

    class Table:
        def __init__(self, header, cells):
            captured["cells"] = cells

    class Figure:
        def __init__(self, data):
            self.data = data

    import sys

    monkeypatch.setitem(sys.modules, "plotly", object())
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", SimpleNamespace(Table=Table, Figure=Figure))

    from mil.tools.features import FeatureRanking
    from mil.viz import feature_table

    ranking = FeatureRanking("sae-x", "h", (1, 1), [{"feature_id": 7, "score": 1.2}])
    feature_table(ranking)

    links = captured["cells"]["values"][-1]
    assert "sae-x+feature+7" in links[0]
