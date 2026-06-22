from pathlib import Path

import src.model_assets as ma


def test_models_ready_false_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    assert ma.models_ready() is False


def test_models_ready_true_when_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    for rel in ma.ARTIFACTS:
        path = tmp_path / "models" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert ma.models_ready() is True


def test_ensure_models_skips_download_when_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    for rel in ma.ARTIFACTS:
        path = tmp_path / "models" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    called = False

    def _fail(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not download")

    monkeypatch.setattr(ma, "hf_hub_download", _fail)
    ma.ensure_models.cache_clear()
    out = ma.ensure_models()
    assert out == tmp_path / "models"
    assert called is False


def test_ensure_models_downloads_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    def _fake_download(repo_id, filename, repo_type):
        assert repo_id == ma.HF_REPO
        assert repo_type == "model"
        stub = tmp_path / "cache" / filename
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(b"payload")
        return str(stub)

    monkeypatch.setattr(ma, "hf_hub_download", _fake_download)
    ma.ensure_models.cache_clear()
    out = ma.ensure_models()
    assert out == models_dir
    for rel in ma.ARTIFACTS:
        assert (models_dir / rel).read_bytes() == b"payload"