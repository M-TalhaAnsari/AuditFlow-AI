
import json
import pytest
from retrieval_evaluation import load_eval_questions


def test_loads_plain_list(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"natural_question": "q1"}, {"natural_question": "q2"}]))
    result = load_eval_questions(str(path))
    assert len(result) == 2


def test_loads_wrapper_key_shape(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"questions": [{"natural_question": "q1"}]}))
    result = load_eval_questions(str(path))
    assert len(result) == 1


def test_loads_difficulty_grouped_shape(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({
        "easy": [{"cuad_question": "q1"}],
        "medium": [{"cuad_question": "q2"}, {"cuad_question": "q3"}],
    }))
    result = load_eval_questions(str(path))
    assert len(result) == 3
    assert {item["difficulty"] for item in result} == {"easy", "medium"}


def test_loads_index_keyed_dict_shape(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"0": {"natural_question": "q1"}, "1": {"natural_question": "q2"}}))
    result = load_eval_questions(str(path))
    assert len(result) == 2
    assert result[0]["natural_question"] == "q1"


def test_unrecognized_dict_shape_raises_clear_error(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"some_key": "not a list", "another_key": 42}))
    with pytest.raises(ValueError, match="doesn't match any known shape"):
        load_eval_questions(str(path))