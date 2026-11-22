from mpa.study import ground_truth, score_prediction


def test_ground_truth_for_demo_prompts():
    assert ground_truth("Can you suggest something I could make for dinner tonight?") == \
        {"scenario": "S1-dinner", "fits": ["m3", "m4"], "misapplied": []}
    assert ground_truth("Can you write a birthday message for my best friend Sam? "
                        "Something warm and a bit funny.")["misapplied"] == ["m2"]
    assert ground_truth("Tell me a joke") is None


def test_prediction_scoring():
    assert score_prediction({"m3", "m4"}, {"m3", "m4"}) == {"exact": True, "jaccard": 1.0}
    assert score_prediction({"m3"}, {"m3", "m4"}) == {"exact": False, "jaccard": 0.5}
    assert score_prediction(set(), set()) == {"exact": True, "jaccard": 1.0}     # "none of them", correctly
    assert score_prediction({"m1"}, set()) == {"exact": False, "jaccard": 0.0}
