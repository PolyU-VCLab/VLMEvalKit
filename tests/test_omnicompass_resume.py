import json
from types import SimpleNamespace

import pandas as pd
import pytest

from vlmeval.dataset.omnicompass import OmniCompass
from vlmeval.inference_video import _reusable_prediction
from vlmeval.smp.file import dump, get_infer_aux_file_names, get_pred_file_path, load


@pytest.fixture
def dataset():
    result = OmniCompass.__new__(OmniCompass)
    result.protocol = SimpleNamespace(protocol_id='fixture')
    result.nframe = 32
    result.fps = -1
    result.docs = [{
        'question_id': 'fixture_tg',
        'setting': setting,
        'protocol_id': 'fixture',
        'answer': '',
    } for setting in ('text', 'text_image')]
    return result


@pytest.mark.parametrize('status,body,reusable', [
    (200, {
        'choices': [{
            'message': {
                'content': 'A'
            }
        }]
    }, True),
    (200, {
        'choices': [{
            'message': {
                'content': ''
            }
        }]
    }, True),
    (400, {
        'error': {
            'code': 'content_policy_violation'
        }
    }, True),
    (504, {
        'error': {
            'code': 'timeout'
        }
    }, False),
])
def test_structured_failure_reuse(dataset, status, body, reusable):
    value = json.dumps({'api_response': {'status_code': status, 'body': body}})
    assert _reusable_prediction(dataset, value) == reusable
    assert not _reusable_prediction(dataset, 'Failed to obtain answer via API.')


def test_reuse_copies_protocol_guard(dataset):
    files = get_infer_aux_file_names('model', 'OmniCompass_VL_32frame', dataset, 'result.xlsx')
    assert dataset.inference_cache_files('model', 'OmniCompass_VL_32frame') <= files


def test_result_only_reuse_handles_excel_empty_cells(dataset, tmp_path):
    path = tmp_path / 'result.xlsx'
    pd.DataFrame([{**d, 'prediction': 'A'} for d in dataset.docs]).to_excel(path, index=False)
    dataset.validate_inference_cache(tmp_path, 'model', path, 'OmniCompass_VL')
    dataset.docs[0]['question_id'] = 'changed'
    with pytest.raises(ValueError, match='configuration changed'):
        dataset.validate_inference_cache(tmp_path, 'other', path, 'OmniCompass_VL')


def test_unverified_partial_checkpoint_rejected(dataset, tmp_path):
    (tmp_path / 'model_OmniCompass_VL_checkpoint.pkl').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='Unverified legacy checkpoints'):
        dataset.validate_inference_cache(tmp_path, 'model', tmp_path / 'result.xlsx', 'OmniCompass_VL')


def test_unverified_rank_checkpoint_rejected(dataset, tmp_path):
    (tmp_path / '01_model_OmniCompass_VL.pkl').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='Unverified legacy checkpoints'):
        dataset.validate_inference_cache(tmp_path, 'model', tmp_path / 'result.tsv', 'OmniCompass_VL')


def test_empty_saved_cell_is_completed(dataset):
    assert _reusable_prediction(dataset, "")
    assert _reusable_prediction(dataset, float("nan"))


def test_generation_changes_reject_resume(dataset, tmp_path):
    model = SimpleNamespace(model="qwen",
                            enable_thinking=True,
                            max_tokens=32768,
                            key="never-save-this",
                            default_kwargs={"reasoning_effort": "high"})
    identity = dataset.inference_model_config(model)
    assert "never-save-this" not in json.dumps(identity)
    args = (tmp_path, "model", tmp_path / "result.tsv", "OmniCompass_VL")
    dataset.validate_inference_cache(*args, generation_config=identity)
    dataset.validate_inference_cache(*args, generation_config=identity)
    for key, value in (("enable_thinking", False), ("max_tokens", 8192)):
        with pytest.raises(ValueError, match="configuration changed"):
            dataset.validate_inference_cache(*args, generation_config={**identity, key: value})


def test_long_generation_roundtrip_and_default_format(tmp_path, monkeypatch):
    monkeypatch.delenv("PRED_FORMAT", raising=False)
    path = get_pred_file_path(str(tmp_path), "qwen", "OmniCompass_VL_32frame")
    assert path.endswith(".tsv")
    assert get_pred_file_path(str(tmp_path), "qwen", "OtherDataset").endswith(".xlsx")
    raw = "<think>" + "x" * 70000 + "</think>C"
    dump(pd.DataFrame([{"prediction": raw}, {"prediction": ""}]), path)
    saved = load(path)
    assert saved.iloc[0]["prediction"] == raw
    monkeypatch.setenv("PRED_FORMAT", "json")
    assert get_pred_file_path(str(tmp_path), "qwen", "OmniCompass_VL").endswith(".json")


def test_native_weight_object_is_not_serialized(dataset, tmp_path):
    model = SimpleNamespace(model=object(), model_path="/fixed/qwen", enable_thinking=False, max_new_tokens=32768)
    identity = dataset.inference_model_config(model)
    assert "model" not in identity
    assert identity["model_path"] == "/fixed/qwen"
    args = (tmp_path, "qwen", tmp_path / "result.tsv", "OmniCompass_VL")
    dataset.validate_inference_cache(*args, generation_config=identity)
    model.model = object()
    dataset.validate_inference_cache(*args, generation_config=dataset.inference_model_config(model))
    model.model_path = "/different/qwen"
    with pytest.raises(ValueError, match="configuration changed"):
        dataset.validate_inference_cache(*args, generation_config=dataset.inference_model_config(model))


def test_api_model_name_is_preserved(dataset):
    identity = dataset.inference_model_config(SimpleNamespace(model="api-qwen"))
    assert json.loads(json.dumps(identity))["model"] == "api-qwen"


def test_replaced_weights_and_declared_revision_invalidate(dataset, tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"weight-v1")
    model = SimpleNamespace(model=object(), model_path=str(weights))
    identity = dataset.inference_model_config(model)
    (weights / "model.safetensors").write_bytes(b"weight-v2")
    assert dataset.inference_model_config(model) != identity
    api = SimpleNamespace(model="api-model")
    monkeypatch.setenv("OMNICOMPASS_MODEL_REVISION", "deployment-v1")
    identity = dataset.inference_model_config(api)
    monkeypatch.setenv("OMNICOMPASS_MODEL_REVISION", "deployment-v2")
    assert dataset.inference_model_config(api) != identity


def test_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    import vlmeval.inference_video as inference

    path = tmp_path / "checkpoint.pkl"
    inference._checkpoint({0: "A"}, path)
    before = path.read_bytes()

    def fail(data, name, **kwargs):
        from pathlib import Path
        Path(name).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(inference, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        inference._checkpoint({0: "A", 1: "B"}, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_runtime_change_rejects_resume(dataset, tmp_path, monkeypatch):
    import vlmeval.dataset.omnicompass as adapter

    model = SimpleNamespace(model="api-model")
    args = (tmp_path, "model", tmp_path / "result.tsv", "OmniCompass_VL")
    dataset.validate_inference_cache(*args, generation_config=dataset.inference_model_config(model))
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {"packages": {"torch": "changed"}})
    with pytest.raises(ValueError, match="configuration changed"):
        dataset.validate_inference_cache(*args, generation_config=dataset.inference_model_config(model))
