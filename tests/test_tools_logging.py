from __future__ import annotations


def test_logged_writes_stderr_on_logging_failure(capsys):
    from mil.tools import logged

    def boom_backend():
        raise RuntimeError("no backend")

    import mil

    mil._get_backend = boom_backend

    @logged
    def f():
        return 3

    assert f() == 3
    assert "mil logging failed in f" in capsys.readouterr().err

