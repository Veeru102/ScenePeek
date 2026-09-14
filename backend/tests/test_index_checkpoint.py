"""index_chunk checkpoints the Whisper transcript so retries don't re-run ASR."""

import json
from types import SimpleNamespace

from scenepeek.ml.whisper import UtteranceOut, Word
from scenepeek.pipeline import index as idx_mod


class _FakeStorage:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def object_exists(self, key):
        return key in self.objects, len(self.objects.get(key, b""))

    def upload_bytes(self, data, key, content_type):
        self.objects[key] = data

    def internal_client(self):
        objs = self.objects
        return SimpleNamespace(
            get_object=lambda Bucket, Key: {"Body": SimpleNamespace(read=lambda: objs[Key])}
        )

    def bucket(self):
        return "b"

    def audio_key(self, video_id):
        return f"videos/{video_id}/audio.wav"


def _fake_transcript(*_a, **_k):
    return [UtteranceOut(60.5, 62.0, "hello world", [Word("hello", 60.5, 61.0), Word("world", 61.2, 62.0)])]


def test_transcript_checkpoint_skips_whisper_on_retry(monkeypatch, tmp_path):
    store = _FakeStorage()
    calls = {"whisper": 0}

    def transcribe(path, offset_s):
        calls["whisper"] += 1
        return _fake_transcript()

    monkeypatch.setattr(idx_mod, "storage", store)
    monkeypatch.setattr(idx_mod.whisper, "transcribe", transcribe)
    monkeypatch.setattr(idx_mod.media, "cached", lambda *a, **k: tmp_path / "audio.wav")
    monkeypatch.setattr(idx_mod.media, "workdir", lambda *a: tmp_path)
    monkeypatch.setattr(idx_mod.media, "slice_audio", lambda *a, **k: None)
    ctx = SimpleNamespace(log=SimpleNamespace(info=lambda *a, **k: None))

    first = idx_mod._transcribe_chunk(ctx, "vid", 1, 60.0, 120.0)
    assert calls["whisper"] == 1
    assert first[0].text == "hello world"
    assert idx_mod.transcript_key("vid", 1) in store.objects

    second = idx_mod._transcribe_chunk(ctx, "vid", 1, 60.0, 120.0)
    assert calls["whisper"] == 1, "retry must reuse the checkpoint instead of re-running Whisper"
    assert [w.w for w in second[0].words] == ["hello", "world"]
    assert second[0].start == first[0].start


def test_transcript_checkpoint_ignored_for_other_model(monkeypatch, tmp_path):
    store = _FakeStorage()
    stale = {
        "model": "some-other-model",
        "utterances": [{"start": 0, "end": 1, "text": "stale", "words": []}],
    }
    store.objects[idx_mod.transcript_key("vid", 0)] = json.dumps(stale).encode()
    calls = {"whisper": 0}

    def transcribe(path, offset_s):
        calls["whisper"] += 1
        return _fake_transcript()

    monkeypatch.setattr(idx_mod, "storage", store)
    monkeypatch.setattr(idx_mod.whisper, "transcribe", transcribe)
    monkeypatch.setattr(idx_mod.media, "cached", lambda *a, **k: tmp_path / "audio.wav")
    monkeypatch.setattr(idx_mod.media, "workdir", lambda *a: tmp_path)
    monkeypatch.setattr(idx_mod.media, "slice_audio", lambda *a, **k: None)
    ctx = SimpleNamespace(log=SimpleNamespace(info=lambda *a, **k: None))

    out = idx_mod._transcribe_chunk(ctx, "vid", 0, 60.0, 120.0)
    assert calls["whisper"] == 1 and out[0].text == "hello world"
