from __future__ import annotations

import pytest


def test_make_parse_local_ref():
    from mil.refs import make_local_ref, parse_ref

    ref = make_local_ref("abcdef1234567890", "acts")
    assert ref == "local:acts:abcdef123456"
    assert parse_ref(ref) == ("local", "acts:abcdef123456")


def test_parse_ref_rejects_bad_format():
    from mil.refs import parse_ref

    with pytest.raises(ValueError):
        parse_ref("not-a-ref")

