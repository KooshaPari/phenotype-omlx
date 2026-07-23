from niah_benchmark import extract_answer_text


def test_extract_answer_text_removes_qwen_control_suffix() -> None:
    answer = "\n</think>\n\n214-125-alpha.<|endoftext|><|im_start|>\n<think>"
    assert extract_answer_text(answer) == "214-125-alpha"


def test_extract_answer_text_does_not_search_inside_explanation() -> None:
    answer = "The critical fact is 214-125-alpha."
    assert extract_answer_text(answer) == "The critical fact is 214-125-alpha"
