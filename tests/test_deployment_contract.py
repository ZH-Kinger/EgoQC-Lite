import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_gpu_student_deployment_contract_is_consistent() -> None:
    contract_path = ROOT / "config" / "qc_student_deployment_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    experiments = json.loads(
        (ROOT / "config" / "qc_ieee_experiments.json").read_text(encoding="utf-8")
    )

    assert contract["schema_version"] == "egoqc-student-deployment-v1"
    assert contract["architecture"]["shared_weights_across_backends"] is True
    assert contract["architecture"]["preferred_parameter_range"] == [16_000_000, 24_000_000]
    assert contract["architecture"]["maximum_parameters"] <= 24_000_000
    assert contract["profiles"]["cpu_int8"]["precision"].startswith("int8")
    assert contract["profiles"]["gpu_fp16"]["precision"] == "fp16"
    assert contract["accuracy_gates"]["abstention_required"] is True
    assert contract["accuracy_gates"]["forced_binary_decision"] is False
    assert contract["benchmark_contract"]["runtime_parity_uses_identical_canonical_input"] is True
    assert contract["accuracy_gates"]["threshold_reselection_on_test_forbidden"] is True
    assert (
        experiments["student_system"]["deployment_contract"]
        == "config/qc_student_deployment_v1.json"
    )
    assert experiments["student_system"]["shared_cpu_gpu_weights"] is True
