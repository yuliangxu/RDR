"""Input acquisition verifies content before installation and preserves inputs."""

import gzip
import hashlib
import io

import pytest

from experiments.MNIST import inputs


def item(content, compressed=False):
    return inputs.Input("MNIST/raw/example", "https://example.invalid/data",
                        hashlib.sha256(content).hexdigest(), compressed)


def no_download(*args, **kwargs):
    pytest.fail("This operation must not access the network")


@pytest.mark.parametrize("compressed", [False, True])
def test_download_verifies_raw_content_and_installs_complete_file(tmp_path, monkeypatch, compressed):
    raw = bytes(range(256)) * 10000
    spec = item(raw, compressed)
    encoded = gzip.compress(raw) if compressed else raw
    monkeypatch.setattr(inputs, "urlopen", lambda *a, **kw: io.BytesIO(encoded))
    record = inputs.acquire(tmp_path, spec)
    assert record["status"] == "downloaded"
    assert (tmp_path / spec.relative).read_bytes() == raw
    assert not list(tmp_path.rglob("*.partial"))
    monkeypatch.setattr(inputs, "urlopen", no_download)
    assert inputs.acquire(tmp_path, spec)["status"] == "verified"
    assert inputs.acquire(tmp_path, spec, verify_only=True)["status"] == "verified"


def test_wrong_download_never_installs_and_cleans_temporary_file(tmp_path, monkeypatch):
    spec = item(b"expected")
    monkeypatch.setattr(inputs, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inputs.acquire(tmp_path, spec)
    assert not (tmp_path / spec.relative).exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_existing_mismatched_file_is_preserved(tmp_path, monkeypatch):
    spec = item(b"expected")
    target = tmp_path / spec.relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing research input")
    monkeypatch.setattr(inputs, "urlopen", no_download)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inputs.acquire(tmp_path, spec)
    assert target.read_bytes() == b"existing research input"
    assert not list(tmp_path.rglob("*.partial"))


def test_verify_missing_input_is_read_only(tmp_path, monkeypatch):
    root = tmp_path / "absent"
    monkeypatch.setattr(inputs, "urlopen", no_download)
    with pytest.raises(FileNotFoundError, match="Missing input"):
        inputs.acquire(root, item(b"expected"), verify_only=True)
    assert not root.exists()


def test_malformed_gzip_cleans_temporary_file(tmp_path, monkeypatch):
    spec = item(b"expected", compressed=True)
    monkeypatch.setattr(inputs, "urlopen", lambda *a, **kw: io.BytesIO(b"not gzip"))
    with pytest.raises(gzip.BadGzipFile):
        inputs.acquire(tmp_path, spec)
    assert not (tmp_path / spec.relative).exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_concurrent_conflicting_input_is_not_overwritten(tmp_path, monkeypatch):
    spec = item(b"expected")
    target = tmp_path / spec.relative
    monkeypatch.setattr(inputs, "urlopen", lambda *a, **kw: io.BytesIO(b"expected"))
    original_link = inputs.os.link

    def race(source, destination):
        target.write_bytes(b"concurrent input")
        original_link(source, destination)

    monkeypatch.setattr(inputs.os, "link", race)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inputs.acquire(tmp_path, spec)
    assert target.read_bytes() == b"concurrent input"
    assert not list(tmp_path.rglob("*.partial"))
