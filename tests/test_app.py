"""Unit tests for app.py.

app.py loads Whisper and parses sys.argv at import time, so we import it once
with Whisper's loader stubbed and --no-tts (which skips Kokoro). No model
weights are loaded and no Ollama server is contacted.
"""

import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _import_app():
    with patch("whisper.load_model", return_value=MagicMock()), \
            patch("sys.argv", ["app.py", "--no-tts"]):
        sys.modules.pop("app", None)
        return importlib.import_module("app")


app = _import_app()


class TestCreateLlm(unittest.TestCase):
    """Tests for the create_llm() factory function."""

    @patch("app.OllamaLLM")
    def test_ollama_provider_default_model(self, mock_ollama):
        """Ollama provider uses default model 'coach'."""
        app.create_llm("ollama")
        mock_ollama.assert_called_once_with(model="coach", base_url="http://localhost:11434")

    @patch("app.OllamaLLM")
    def test_ollama_provider_custom_model(self, mock_ollama):
        """Ollama provider accepts a custom model name."""
        app.create_llm("ollama", model="llama3")
        mock_ollama.assert_called_once_with(model="llama3", base_url="http://localhost:11434")

    def test_unknown_provider_raises(self):
        """Unknown provider raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            app.create_llm("unknown_provider")
        self.assertIn("Unknown provider", str(ctx.exception))

    def test_cloud_provider_rejected(self):
        """Previously-supported cloud provider names are no longer accepted."""
        with self.assertRaises(ValueError):
            app.create_llm("minimax")


class TestStreamSentences(unittest.TestCase):
    """Tests for stream_sentences(), which splits streamed LLM chunks into sentences."""

    def test_splits_on_terminal_punctuation(self):
        chunks = ["Hello there. How ", "are you? I'm ", "fine!"]
        self.assertEqual(
            list(app.stream_sentences(chunks)),
            ["Hello there.", "How are you?", "I'm fine!"],
        )

    def test_sentence_spanning_chunks(self):
        chunks = ["Th", "is is o", "ne sentence."]
        self.assertEqual(list(app.stream_sentences(chunks)), ["This is one sentence."])

    def test_trailing_partial_text_yielded_last(self):
        chunks = ["Done. And then"]
        self.assertEqual(list(app.stream_sentences(chunks)), ["Done.", "And then"])

    def test_decimal_point_does_not_split(self):
        chunks = ["It costs 3.50 today."]
        self.assertEqual(list(app.stream_sentences(chunks)), ["It costs 3.50 today."])

    def test_empty_and_whitespace_chunks_ignored(self):
        self.assertEqual(list(app.stream_sentences(["", None, "   "])), [])


class TestStripMarkdownForTts(unittest.TestCase):
    """Tests for strip_markdown_for_tts(), which keeps Kokoro from reading markup aloud."""

    def test_removes_emphasis_markers(self):
        self.assertEqual(
            app.strip_markdown_for_tts("This is **bold**, *italic* and `code`."),
            "This is bold, italic and code.",
        )

    def test_removes_stray_markdown_characters(self):
        self.assertEqual(app.strip_markdown_for_tts("## Heading"), " Heading")
        self.assertEqual(app.strip_markdown_for_tts("> quoted"), " quoted")

    def test_plain_text_unchanged(self):
        text = "Nothing to strip here, right?"
        self.assertEqual(app.strip_markdown_for_tts(text), text)


class TestStreamLlmToText(unittest.TestCase):
    """Tests for stream_llm_to_text() and how it records conversation history."""

    def setUp(self):
        app.chat_history.clear()

    def test_returns_joined_response_and_records_history(self):
        fake_chain = MagicMock()
        fake_chain.stream.return_value = iter(["Hi ", "there. ", ""])
        with patch.object(app, "chain", fake_chain), patch.object(app, "console"):
            result = app.stream_llm_to_text("hello")

        self.assertEqual(result, "Hi there.")
        self.assertEqual(
            [(type(m).__name__, m.content) for m in app.chat_history],
            [("HumanMessage", "hello"), ("AIMessage", "Hi there.")],
        )

    def test_prior_turns_passed_as_history(self):
        fake_chain = MagicMock()
        fake_chain.stream.side_effect = [iter(["First."]), iter(["Second."])]
        with patch.object(app, "chain", fake_chain), patch.object(app, "console"):
            app.stream_llm_to_text("one")
            app.stream_llm_to_text("two")

        second_call = fake_chain.stream.call_args_list[1].args[0]
        self.assertEqual(second_call["input"], "two")
        self.assertEqual([m.content for m in second_call["history"]], ["one", "First."])


if __name__ == "__main__":
    unittest.main()
