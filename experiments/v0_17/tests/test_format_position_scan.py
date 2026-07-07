from run_format_position_scan import PROMPT_FORMATS, parse_format_list


def test_parse_format_list() -> None:
    assert parse_format_list("task_response,user_assistant") == ["task_response", "user_assistant"]


def test_prompt_formats_include_response_boundaries() -> None:
    assert PROMPT_FORMATS["task_response"].endswith("Response:")
    assert PROMPT_FORMATS["instruction_answer"].endswith("Answer:")
    assert PROMPT_FORMATS["user_assistant"].endswith("Assistant:")
