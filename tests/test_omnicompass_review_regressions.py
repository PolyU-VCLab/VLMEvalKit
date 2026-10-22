"""Regression coverage for the 2026-10-22 independent integration review."""

import hashlib
import json
import os
import pickle
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from vlmeval.api.gpt import OpenAIWrapper
from vlmeval.config import supported_VLM
from vlmeval.dataset.omnicompass import OmniCompass
from vlmeval.dataset.utils.omnicompass.bridge import evaluate, prediction
from vlmeval.dataset.utils.omnicompass.run_identity import model_identity
from vlmeval.inference_video import _build_video_model, _checkpoint, infer_data_job_video
from vlmeval.smp import dump, load
from vlmeval.vlm.base import BaseModel
from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat


class TinyDataset(OmniCompass):

    def __init__(self, task="QA"):
        self.dataset_name = "OmniCompass_VL"
        self.settings = ("text", )
        self.protocol = SimpleNamespace(protocol_id="review-regression")
        self.nframe, self.fps = 32, -1
        ground_truth = {
            "QA": {
                "answer": "A"
            },
            "TG": {
                "start_time": 1,
                "end_time": 2
            },
            "SG": {
                "start_bbox": [100, 200, 500, 600],
                "end_bbox": [100, 200, 500, 600]
            },
        }[task]
        question = {
            "question_id": "fixture",
            "video_id": "fixture",
            "track": "VL",
            "task": task,
            "ground_truth": ground_truth
        }
        if task == "SG":
            question["given_interval"] = [1, 2]
        self.docs = [{
            "index": 0,
            "question_id": "fixture",
            "video": "fixture",
            "track": "VL",
            "task": task,
            "setting": "text",
            "question_json": json.dumps(question),
            "protocol_id": self.protocol.protocol_id,
            "answer": "",
        }]
        self.data = pd.DataFrame(self.docs)

    def build_prompt(self, line, video_llm=False):
        return [{"type": "text", "value": "Offline fixture"}]


def api_model(**kwargs):
    return OpenAIWrapper(model="fixture",
                         key="offline-placeholder",
                         api_base="http://127.0.0.1:1/v1/chat/completions",
                         retry=1,
                         wait=0,
                         **kwargs)


@pytest.mark.parametrize("field,value", [
    ("img_size", 512),
    ("total_img_size", 1024),
    ("max_file_size", 1000),
])
def test_input_change_rejects_completed_cache(tmp_path, field, value):
    model, dataset = api_model(), TinyDataset()
    model.generate = Mock(return_value="A")
    args = (model, str(tmp_path), "fixture", dataset)
    result = str(tmp_path / "result.tsv")
    infer_data_job_video(*args, result_file=result, api_nproc=1)
    model.generate.reset_mock()
    setattr(model, field, value)
    with pytest.raises(ValueError, match="configuration changed"):
        infer_data_job_video(*args, result_file=result, api_nproc=1)
    model.generate.assert_not_called()
    assert load(result).prediction.tolist() == ["A"]


@pytest.mark.parametrize("name,value", [
    ("seed", 2),
    ("frequency_penalty", 0.5),
    ("response_format", {
        "type": "json_object"
    }),
    ("chat_template_kwargs", {
        "custom_option": True
    }),
])
def test_all_provider_parameters_invalidate(tmp_path, name, value):
    model = api_model(seed=1)
    dataset = TinyDataset()
    args = (tmp_path, "fixture", tmp_path / "result.tsv", "OmniCompass_VL")
    original = dataset.inference_model_config(model)
    dataset.validate_inference_cache(*args, generation_config=original)
    model.default_kwargs[name] = value
    with pytest.raises(ValueError, match="configuration changed"):
        dataset.validate_inference_cache(*args, generation_config=dataset.inference_model_config(model))
    assert "offline-placeholder" not in json.dumps(original)


def test_provider_seed_reaches_payload_and_changes_identity(monkeypatch):
    model, seen = api_model(seed=1), []

    def post(*args, **kwargs):
        seen.append(json.loads(kwargs["data"]))
        return SimpleNamespace(status_code=200,
                               headers={},
                               encoding="utf-8",
                               json=lambda: {"choices": [{
                                   "message": {
                                       "content": "A"
                                   }
                               }]})

    monkeypatch.setattr("vlmeval.api.gpt.requests.post", post)
    before = OmniCompass.inference_model_config(model)
    model.generate_inner([{"type": "text", "value": "fixture"}], **model.default_kwargs)
    model.default_kwargs["seed"] = 2
    model.generate_inner([{"type": "text", "value": "fixture"}], **model.default_kwargs)
    assert [p["seed"] for p in seen] == [1, 2]
    assert before != OmniCompass.inference_model_config(model)


@pytest.mark.parametrize("routing", [{}, {
    "dataset": None
}, {
    "dataset": "OtherBenchmark"
}, {
    "dataset": "OmniCompass_VL"
}, {
    "dataset": "OmniCompass_AV"
}])
def test_api_dataset_payload_scope(monkeypatch, routing):
    model = api_model()
    response = SimpleNamespace(status_code=200,
                               headers={},
                               encoding="utf-8",
                               text='{"choices": [{"message": {"content": "A"}}]}',
                               json=lambda: {"choices": [{
                                   "message": {
                                       "content": "A"
                                   }
                               }]})
    post = Mock(return_value=response)
    monkeypatch.setattr("vlmeval.api.gpt.requests.post", post)
    code, answer, _ = model.generate_inner([{"type": "text", "value": "fixture"}], **routing)
    assert (code, answer) == (0, "A")
    expected = {
        "model": "fixture",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "fixture"
            }]
        }],
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
    }
    if not (routing.get("dataset") or "").startswith("OmniCompass"):
        expected.update(routing)
    assert json.loads(post.call_args.kwargs["data"]) == expected


@pytest.mark.parametrize("dataset", [None, "OtherBenchmark", "OmniCompass_VL"])
def test_api_system_prompt_is_request_scoped(monkeypatch, dataset):
    model = api_model(system_prompt="default")
    response = SimpleNamespace(status_code=200,
                               headers={},
                               encoding="utf-8",
                               text='{"choices": [{"message": {"content": "A"}}]}')
    post = Mock(return_value=response)
    monkeypatch.setattr("vlmeval.api.gpt.requests.post", post)
    first = [
        {"type": "text", "value": "request-only", "role": "system"},
        {"type": "text", "value": "first"},
    ]
    assert model.generate(first, dataset=dataset) == "A"
    assert model.generate("second", dataset=dataset) == "A"
    payloads = [json.loads(call.kwargs["data"]) for call in post.call_args_list]
    assert [p["messages"][0] for p in payloads] == [
        {"role": "system", "content": "default\nrequest-only"},
        {"role": "system", "content": "default"},
    ]
    assert all("system_prompt" not in payload for payload in payloads)
    assert model.system_prompt == "default"


def test_endpoint_and_dataset_order_with_real_factory(tmp_path, monkeypatch):
    monkeypatch.setitem(supported_VLM, "review-api",
                        partial(OpenAIWrapper, model="fixture", key="offline-placeholder", retry=1, wait=0))
    generate = Mock(return_value="A")
    monkeypatch.setattr(OpenAIWrapper, "generate", generate)
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:1/deployment-a")

    def run(model, alias):
        return infer_data_job_video(model,
                                    str(tmp_path),
                                    "review-api",
                                    TinyDataset(),
                                    result_file=str(tmp_path / f"{alias}.tsv"),
                                    dataset_alias_name=alias,
                                    api_nproc=1)

    # The CLI uses a string for A, then the returned object for B.
    model = run("review-api", "VL_A")
    assert run(model, "VL_B") is model
    assert generate.call_count == 2
    # Resume B alone, then reverse the dataset order.
    model = run("review-api", "VL_B")
    run(model, "VL_A")
    assert generate.call_count == 2
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:1/deployment-b")
    with pytest.raises(ValueError, match="configuration changed"):
        run("review-api", "VL_B")
    assert generate.call_count == 2


def test_constructor_failure_restores_distributed_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    factory = Mock(side_effect=RuntimeError("fixture init failed"))
    monkeypatch.setitem(supported_VLM, "fixture", factory)
    with pytest.raises(RuntimeError, match="init failed"):
        _build_video_model("fixture", "fixture", False)
    assert os.environ["WORLD_SIZE"] == "2"


@pytest.mark.parametrize("suffix", ["tsv", "csv", "json", "xlsx", "pkl"])
def test_checkpoint_format_roundtrip(tmp_path, suffix):
    path = tmp_path / f"result.{suffix}"
    _checkpoint(pd.DataFrame([{"index": 0, "prediction": "old"}]), path)
    _checkpoint(pd.DataFrame([{"index": 1, "prediction": "new"}]), path)
    saved = load(str(path))
    saved = saved if isinstance(saved, pd.DataFrame) else pd.DataFrame(saved)
    assert saved.prediction.tolist() == ["new"]
    assert list(tmp_path.iterdir()) == [path]


def test_serializer_failure_keeps_target_and_default_dump_fallback(tmp_path, monkeypatch):
    path = tmp_path / "result.tsv"
    frame = pd.DataFrame([{"prediction": "old"}])
    _checkpoint(frame, path)
    original = path.read_bytes()
    monkeypatch.setattr(pd.DataFrame, "to_csv", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        _checkpoint(frame, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    dump(frame, str(tmp_path / "legacy.tsv"))
    assert load(str(tmp_path / "legacy.pkl")).prediction.tolist() == ["old"]


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX RLIMIT_FSIZE")
def test_real_kernel_write_failure_preserves_target(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, "--write-limit", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=45,
        check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRESERVED" in result.stdout


def write_limit_worker(root):
    import resource
    import signal

    path = root / "result.tsv"
    _checkpoint(pd.DataFrame([{"prediction": "old"}]), path)
    original = path.read_bytes()
    # Quoted JSON grows more in TSV than in pickle, matching real response records.
    frame = pd.DataFrame({
        "prediction": [
            json.dumps({
                "api_response": {
                    "body": {
                        "choices": [{
                            "message": {
                                "content": "A",
                                "raw_generation": str(i) + '"x",' * 500
                            }
                        }]
                    }
                }
            }) for i in range(400)
        ]
    })
    pkl_size = len(pickle.dumps(frame))
    tsv_size = len(frame.to_csv(sep="\t", index=False).encode())
    assert pkl_size < tsv_size
    limit = (pkl_size + tsv_size) // 2
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    old_limits = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, old_limits[1]))
    try:
        with pytest.raises(OSError):
            _checkpoint(frame, path)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, old_limits)
    assert path.read_bytes() == original
    assert list(root.iterdir()) == [path]
    print(f"PRESERVED pickle={pkl_size} limit={limit} tsv={tsv_size}")


@pytest.mark.parametrize("task,answer", [
    ("QA", "A"),
    ("TG", "[1, 2]"),
    ("SG", ('{"start_time": 1, "end_time": 2, "start_bbox": [100, 200, 500, 600], '
            '"end_bbox": [100, 200, 500, 600]}')),
])
@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("state", ["plain", "closed", "truncated"])
def test_only_final_answer_scores(task, answer, envelope, state):
    dataset = TinyDataset(task)
    raw = {
        "plain": answer,
        "closed": f"<think>analysis</think>{answer}",
        "truncated": f"<think>considering {answer}"
    }[state]
    output = json.dumps({
        "api_response": {
            "status_code": 200,
            "body": {
                "choices": [{
                    "message": {
                        "content": raw
                    },
                    "finish_reason": "length"
                }]
            }
        }
    }) if envelope else raw
    row = prediction(dataset.docs[0], output)
    assert row["raw_output"] == ("" if state == "truncated" else answer)
    if state != "plain" or envelope:
        assert row["response_meta"]["raw_generation"] == raw
    report = evaluate(dataset.docs, [output], settings=("text", ))
    assert report["parsed_predictions"][0]["score"] == (0 if state == "truncated" else 1)


@pytest.mark.parametrize("status,body,expected", [
    (504, {
        "error": {
            "code": "timeout"
        }
    }, "transport"),
    (400, {
        "error": {
            "code": "content_policy_violation"
        }
    }, "content_policy"),
])
def test_errors_remain_excluded(status, body, expected):
    row = prediction({
        "question_id": "q",
        "setting": "text"
    }, json.dumps({"api_response": {
        "status_code": status,
        "body": body
    }}))
    assert row["error_kind"] == expected
    assert row["raw_output"] == ""


class Batch(dict):

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, *args):
        return self


def construct_qwen(monkeypatch, path):
    processor = Mock()
    processor.return_value = Batch(input_ids=torch.zeros((1, 4), dtype=torch.long))
    processor.apply_chat_template.return_value = "fixture"
    processor.batch_decode.return_value = ["A"]
    processor.tokenizer.batch_decode.return_value = ["A"]
    weights = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=65536)),
        device="cpu",
        dtype=torch.float32,
        eval=Mock(),
        generate=Mock(return_value=(SimpleNamespace(sequences=torch.zeros((1, 5), dtype=torch.long)), None)),
    )
    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=Mock(return_value=weights)),
            Qwen3OmniMoeProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            Qwen3OmniMoeForConditionalGeneration=SimpleNamespace(from_pretrained=Mock(return_value=weights)),
        ))
    monkeypatch.setattr("vlmeval.vlm.qwen3_vl.model.get_gpu_memory", lambda: [80])
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    obj = Qwen3VLChat(path, use_custom_prompt=False, use_audio_in_video=False)
    return obj


def test_qwen3_omni_constructor_and_separate_audio(monkeypatch):
    original_types = list(BaseModel.allowed_types)
    obj = construct_qwen(monkeypatch, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    dataset = TinyDataset()
    dataset.dataset_name = "OmniCompass_AV"
    dataset.validate_model(obj)
    process = Mock(return_value=(["fixture-audio"], ["fixture-image"], None))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=process))
    # yapf: disable
    result = obj.generate_inner_transformers([
        {
            "type": "audio",
            "value": "fixture.wav"
        },
        {
            "type": "image",
            "value": "file:///fixture.jpg"
        },
        {
            "type": "text",
            "value": "fixture"
        },
    ],
        dataset="OmniCompass_AV")
    # yapf: enable
    assert prediction(dataset.docs[0], result)["raw_output"] == "A"
    assert obj.processor.call_args.kwargs["audio"] == ["fixture-audio"]
    assert obj.processor.call_args.kwargs["images"] == ["fixture-image"]
    assert process.call_args.kwargs["use_audio_in_video"] is False
    assert obj.model.generate.call_args.kwargs["thinker_max_new_tokens"] == 32768
    assert BaseModel.allowed_types == original_types
    ordinary = construct_qwen(monkeypatch, "Qwen/Qwen3-VL-8B")
    assert "audio" not in ordinary.allowed_types
    for flag in ("use_vllm", "use_audio_in_video"):
        setattr(obj, flag, True)
        with pytest.raises(ValueError, match="separate-audio"):
            dataset.validate_model(obj)
        setattr(obj, flag, False)


@pytest.mark.parametrize("dataset,context,should_fail", [
    ("OmniCompass_VL", None, True),
    ("OtherBenchmark", None, False),
    ("OtherBenchmark", 32768, True),
])
def test_context_check_scope(monkeypatch, dataset, context, should_fail):
    obj = construct_qwen(monkeypatch, "Qwen/Qwen3-VL-8B")
    obj.model.config.text_config.max_position_embeddings = 32768
    obj.model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    obj.context_length = context
    monkeypatch.setitem(sys.modules, "qwen_vl_utils",
                        SimpleNamespace(process_vision_info=lambda *a, **k: (None, None, {})))
    if should_fail:
        with pytest.raises(ValueError, match="Context capacity exceeded"):
            obj.generate_inner_transformers([{"type": "text", "value": "fixture"}], dataset=dataset)
        obj.model.generate.assert_not_called()
    else:
        assert obj.generate_inner_transformers([{"type": "text", "value": "fixture"}], dataset=dataset) == "A"


@pytest.mark.parametrize("dataset", [None, "OtherBenchmark", "OmniCompass_AV"])
@pytest.mark.parametrize("fallback", [False, True])
def test_omni_budget_is_scoped_to_omnicompass(monkeypatch, dataset, fallback):
    obj = construct_qwen(monkeypatch, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    monkeypatch.setitem(sys.modules, "qwen_omni_utils",
                        SimpleNamespace(process_mm_info=lambda *args, **kwargs: (None, None, None)))
    if fallback:
        obj.model.generate.side_effect = [TypeError("fixture"), obj.model.generate.return_value]
    obj.generate_inner_transformers([{"type": "text", "value": "fixture"}], dataset=dataset)
    assert obj.model.generate.call_count == (2 if fallback else 1)
    for call in obj.model.generate.call_args_list:
        if dataset == "OmniCompass_AV":
            assert call.kwargs["thinker_max_new_tokens"] == obj.max_new_tokens
        else:
            assert "thinker_max_new_tokens" not in call.kwargs


def test_local_artifact_hash_supports_python310(tmp_path, monkeypatch):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    content = b'{"fixture": true}\n'
    (tmp_path / "config.json").write_bytes(content)
    identity = model_identity(tmp_path)
    assert identity["local_artifacts_sha256"] == {"config.json": hashlib.sha256(content).hexdigest()}


@pytest.mark.parametrize("suffix", ["tsv", "json"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_prediction_table_completion_and_resume(tmp_path, suffix, interrupted):
    model, dataset = api_model(), TinyDataset()
    failed = json.dumps({"api_response": {"status_code": 504, "body": {"error": "timeout"}}})
    model.generate = Mock(side_effect=[failed, "A"] if interrupted else ["A"])
    args = (model, str(tmp_path), "fixture", dataset)
    path = str(tmp_path / f"result.{suffix}")
    infer_data_job_video(*args, result_file=path, api_nproc=1)
    if interrupted:
        with pytest.raises(ValueError, match="Incomplete transport"):
            dataset.validate_completed_predictions(path)
        infer_data_job_video(*args, result_file=path, api_nproc=1)
    dataset.validate_completed_predictions(path)
    assert dataset.evaluate(path)["tracks"]["VL"]["main_metrics"]["text_target"]["QA_accuracy"] == 1.0
    infer_data_job_video(*args, result_file=path, api_nproc=1)
    assert model.generate.call_count == (2 if interrupted else 1)


@pytest.mark.parametrize("backend", ["transformers", "vllm"])
@pytest.mark.parametrize("task,answer", [("QA", "A"), ("TG", "[1, 2]")])
@pytest.mark.parametrize("closed", [False, True])
def test_prefilled_thinking_template_retains_raw_and_scores_final(monkeypatch, backend, task, answer, closed):
    obj = construct_qwen(monkeypatch, "Qwen/Qwen3-VL-8B-Thinking")
    obj.processor.apply_chat_template.return_value = "<|im_start|>assistant\n<think>\n"
    raw = f"Let me think. It might be {answer} but wait"
    if closed:
        raw += f"</think>{answer}"
    obj.processor.tokenizer.batch_decode.return_value = [raw]
    obj.model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils",
                        SimpleNamespace(process_vision_info=lambda *a, **k: (None, None, {})))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **k: k))
    obj.llm = SimpleNamespace(generate=Mock(return_value=[
        SimpleNamespace(prompt_token_ids=[1] * 4, outputs=[SimpleNamespace(text=raw, token_ids=[1])]),
    ]))
    generate = getattr(obj, f"generate_inner_{backend}")
    output = generate([{"type": "text", "value": "fixture"}], dataset="OmniCompass_VL")
    dataset = TinyDataset(task)
    row = prediction(dataset.docs[0], output)
    assert obj.enable_thinking is None
    assert obj.return_response_json is False
    assert row["raw_output"] == (answer if closed else "")
    assert row["response_meta"]["raw_generation"] == raw
    score = evaluate(dataset.docs, [output], settings=("text", ))["parsed_predictions"][0]["score"]
    assert score == (1 if closed else 0)
    # Other datasets keep their original plain string result and parsing.
    assert generate([{"type": "text", "value": "fixture"}], dataset="OtherBenchmark") == raw


def test_xlsx_cannot_truncate_implicit_native_response_records(tmp_path):
    dataset = TinyDataset()
    with pytest.raises(ValueError, match="PRED_FORMAT=tsv"):
        dataset.validate_inference_cache(tmp_path,
                                         "fixture",
                                         tmp_path / "result.xlsx",
                                         "OmniCompass_VL",
                                         generation_config={"return_response_json": False})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_checkpoint_respects_umask_and_existing_mode(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, "--permissions", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=45,
        check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def permissions_worker(root):
    import stat

    old = os.umask(0o027)
    try:
        direct, atomic = root / "direct.pkl", root / "atomic.pkl"
        dump({"answer": "A"}, str(direct))
        _checkpoint({"answer": "A"}, atomic)
        assert stat.S_IMODE(atomic.stat().st_mode) == stat.S_IMODE(direct.stat().st_mode) == 0o640
        atomic.chmod(0o664)
        _checkpoint({"answer": "B"}, atomic)
        assert stat.S_IMODE(atomic.stat().st_mode) == 0o664
        assert load(str(atomic)) == {"answer": "B"}
    finally:
        os.umask(old)


if __name__ == "__main__":
    if sys.argv[1] == "--write-limit":
        write_limit_worker(Path(sys.argv[2]))
    elif sys.argv[1] == "--permissions":
        permissions_worker(Path(sys.argv[2]))
