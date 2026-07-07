from run_chat_template_scan import build_chat_prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "Decide."}]
        assert tokenize is False
        assert add_generation_prompt is True
        return "<user>Decide.</user><assistant>"


def test_build_chat_prompt_uses_generation_boundary():
    assert build_chat_prompt(FakeTokenizer(), "Decide.") == "<user>Decide.</user><assistant>"
