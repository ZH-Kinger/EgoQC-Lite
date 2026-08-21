from egoqc.data_classification import annotation_provenance, classify_capability_profile


def test_capability_profile_keeps_rgb_text_separate_from_mano():
    assert classify_capability_profile({"video": True, "task_labels": True}) == "rgb_text"
    assert classify_capability_profile({"video": True, "mano_parameters": True}) == "rgb_mano"


def test_derived_mano_never_becomes_ground_truth():
    profile = annotation_provenance(
        task_present=True,
        derived_hand_screen_present=True,
        derived_mano_present=True,
        alignment_human_approved=True,
    )
    assert profile["mano"] == "derived_silver_prediction_human_approved"
    assert profile["mano_training_eligible"] is True
    assert profile["mano_is_ground_truth"] is False
