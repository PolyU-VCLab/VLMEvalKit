"""The real CLI dispatcher must propagate incomplete/failed combinations."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import run as runner


def exercise(root, fault, monkeypatch, dataset_name="OmniCompass_VL", config_alias=False):
    args = SimpleNamespace(
        config=None,
        data=[dataset_name],
        model=["fixture"],
        data_config={},
        work_dir=str(root),
        mode="eval" if fault == "eval_missing" else "infer" if fault == "infer_transport" else "all",
        reuse=False,
        reuse_aux="all",
        base_url=None,
        api_nproc=1,
        retry=1,
        verbose=False,
        keep_failed=False,
        use_vllm=False,
        judge_api_nproc=None,
        judge_retry=None,
        judge_timeout=600,
        judge_args=None,
        judge_base_url=None,
        judge_key=None,
        judge=None,
    )
    if config_alias:
        args.config, args.data, args.model = "fixture.json", None, None
        monkeypatch.setattr(
            runner, "load",
            Mock(return_value={
                "model": {
                    "fixture": {}
                },
                "data": {
                    "custom-track": {
                        "class": "OmniCompass",
                        "dataset": dataset_name
                    }
                },
            }))
        monkeypatch.setattr(runner, "build_model_from_config", Mock(return_value=object()))
    dataset = SimpleNamespace(
        data=pd.DataFrame({"index": [0]}),
        MODALITY="VIDEO",
        TYPE="Video-VQA",
        evaluate=Mock(return_value={"fixture": 1}),
        validate_completed_predictions=Mock(),
    )
    if fault == "evaluation":
        dataset.evaluate.side_effect = ValueError("injected scorer failure")
    if fault == "infer_transport":
        dataset.validate_completed_predictions.side_effect = ValueError("Incomplete transport results")
    replacements = {
        "RANK":
        0,
        "WORLD_SIZE":
        1,
        "build_eval_id":
        lambda: "CURRENT",
        "githash":
        lambda **kw: "fixture",
        "setup_logger":
        Mock(),
        "apply_supported_vlm_cli_overrides":
        Mock(),
        "upsert_run_status":
        Mock(),
        "upsert_dataset_status":
        Mock(),
        "log_run_benchmark_report":
        Mock(),
        "get_pred_file_path":
        lambda *a, **kw: str(root / "fixture.tsv"),
        "build_dataset_from_spec":
        Mock(return_value=None if fault == "invalid" else dataset),
        "get_judge_kwargs":
        Mock(return_value={}),
        "prepare_reuse_files":
        Mock(return_value={
            "source_eval_id": None,
            "prediction_complete": False
        }),
        "is_prediction_complete":
        Mock(return_value=False),
        "infer_data_job_video":
        Mock(side_effect=ValueError("injected inference failure") if fault == "inference" else None),
    }
    for name, value in replacements.items():
        monkeypatch.setattr(runner, name, value)
    runner.run_local_mode(args)


@pytest.mark.parametrize("fault", ["invalid", "inference", "evaluation", "eval_missing", "infer_transport"])
def test_failed_combination_has_nonzero_exit(tmp_path, fault):
    result = subprocess.run(
        [sys.executable, __file__, "--worker", str(tmp_path), fault], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Evaluation failed for" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("fault", ["invalid", "inference", "evaluation", "eval_missing", "infer_transport"])
def test_other_benchmarks_keep_original_exit_behavior(tmp_path, fault, monkeypatch):
    exercise(tmp_path, fault, monkeypatch, dataset_name="MMBench-Video")


@pytest.mark.parametrize("fault", ["invalid", "inference", "evaluation", "eval_missing", "infer_transport"])
def test_omnicompass_config_alias_reports_failure(tmp_path, fault, monkeypatch):
    with pytest.raises(RuntimeError, match="Evaluation failed for"):
        exercise(tmp_path, fault, monkeypatch, config_alias=True)


def test_other_benchmark_config_alias_keeps_exit_behavior(tmp_path, monkeypatch):
    exercise(tmp_path, "inference", monkeypatch, dataset_name="MMBench-Video", config_alias=True)


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as monkeypatch:
        exercise(Path(sys.argv[2]), sys.argv[3], monkeypatch)
