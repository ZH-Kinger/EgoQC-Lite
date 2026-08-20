import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_gpu_student_deployment_contract_is_consistent() -> None:
    contract_path = ROOT / "config" / "qc_student_deployment_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    experiments = json.loads(
        (ROOT / "config" / "qc_ieee_experiments.json").read_text(encoding="utf-8")
    )

    assert contract["schema_version"] == "egoqc-student-deployment-v2"
    scout = contract["architecture"]["compact_scout"]
    vlm = contract["architecture"]["primary_vlm"]
    assert scout["preferred_parameter_range"] == [16_000_000, 24_000_000]
    assert scout["decision_authority"] == "route_only"
    assert vlm["preferred_parameter_scale_b"] == 4
    assert vlm["candidate_parameter_scales_b"] == [2, 4, 8]
    assert vlm["training_primary"] == "full_parameter_sft"
    assert contract["architecture"]["shared_base_checkpoint_across_cpu_gpu"] is True
    assert contract["profiles"]["cpu_scout_int8"]["precision"].startswith("int8")
    assert "int4" in contract["profiles"]["cpu_vlm_int4"]["precision"]
    assert contract["profiles"]["gpu_vlm_bf16"]["precision"] == "bf16_or_fp16"
    assert contract["accuracy_gates"]["abstention_required"] is True
    assert contract["accuracy_gates"]["forced_binary_decision"] is False
    assert contract["benchmark_contract"]["runtime_parity_uses_identical_canonical_input"] is True
    assert contract["accuracy_gates"]["threshold_reselection_on_test_forbidden"] is True
    assert (
        experiments["student_system"]["deployment_contract"]
        == "config/qc_student_deployment_v1.json"
    )
    assert experiments["student_system"]["primary_model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert experiments["student_system"]["shared_base_checkpoint_across_cpu_gpu"] is True
    assert len(experiments["student_system"]["scale_ablation"]) == 3
