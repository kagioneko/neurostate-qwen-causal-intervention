from inspect_prompt_tokens import parse_int_list


def test_parse_int_list() -> None:
    assert parse_int_list("-12,-10, -8,-1") == [-12, -10, -8, -1]
