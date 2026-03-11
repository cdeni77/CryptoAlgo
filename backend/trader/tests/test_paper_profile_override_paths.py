from pathlib import Path

from scripts import train_model


def test_default_paper_profile_overrides_path_when_env_unset(monkeypatch):
    monkeypatch.delenv("PAPER_PROFILE_OVERRIDES_PATH", raising=False)
    path = train_model.resolve_paper_profile_overrides_path()
    assert path == Path("data") / "paper_candidates"


def test_env_override_path_wins_for_paper_profile_overrides(monkeypatch):
    monkeypatch.setenv("PAPER_PROFILE_OVERRIDES_PATH", "/tmp/custom-paper-candidates")
    path = train_model.resolve_paper_profile_overrides_path()
    assert path == Path("/tmp/custom-paper-candidates")
