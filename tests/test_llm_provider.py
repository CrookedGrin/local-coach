"""Unit tests for the LLM provider factory in app.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import create_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCreateLlm(unittest.TestCase):
    """Tests for the create_llm() factory function."""

    def _import_create_llm(self):
        """Import create_llm with heavy dependencies mocked out."""
        mock_nltk = MagicMock()
        mock_nltk.__spec__ = MagicMock()  # transformers checks __spec__ to detect nltk

        mock_modules = {
            "whisper": MagicMock(),
            "sounddevice": MagicMock(),
            "numpy": MagicMock(),
            "torch": MagicMock(),
            "torchaudio": MagicMock(),
            "nltk": mock_nltk,
            "chatterbox": MagicMock(),
            "chatterbox.tts": MagicMock(),
            "tts": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            with patch("sys.argv", ["app.py"]):
                if "app" in sys.modules:
                    del sys.modules["app"]
                from app import create_llm
        return create_llm

    @patch("langchain_ollama.OllamaLLM")
    def test_ollama_provider_default_model(self, mock_ollama):
        """Ollama provider uses default model 'coach'."""
        create_llm = self._import_create_llm()
        mock_ollama.reset_mock()  # Clear calls from module-level initialization
        create_llm("ollama")
        mock_ollama.assert_called_once_with(model="coach", base_url="http://localhost:11434")

    @patch("langchain_ollama.OllamaLLM")
    def test_ollama_provider_custom_model(self, mock_ollama):
        """Ollama provider accepts a custom model name."""
        create_llm = self._import_create_llm()
        mock_ollama.reset_mock()
        create_llm("ollama", model="llama3")
        mock_ollama.assert_called_once_with(model="llama3", base_url="http://localhost:11434")

    def test_unknown_provider_raises(self):
        """Unknown provider raises ValueError."""
        create_llm = self._import_create_llm()
        with self.assertRaises(ValueError) as ctx:
            create_llm("unknown_provider")
        self.assertIn("Unknown provider", str(ctx.exception))

    def test_cloud_provider_rejected(self):
        """Previously-supported cloud provider names are no longer accepted."""
        create_llm = self._import_create_llm()
        with self.assertRaises(ValueError):
            create_llm("minimax")


class TestAnalyzeEmotion(unittest.TestCase):
    """Tests for the analyze_emotion() helper function."""

    def _import_analyze_emotion(self):
        mock_modules = {
            "whisper": MagicMock(),
            "sounddevice": MagicMock(),
            "numpy": MagicMock(),
            "torch": MagicMock(),
            "torchaudio": MagicMock(),
            "nltk": MagicMock(),
            "chatterbox": MagicMock(),
            "chatterbox.tts": MagicMock(),
            "tts": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            with patch("sys.argv", ["app.py"]):
                if "app" in sys.modules:
                    del sys.modules["app"]
                from app import analyze_emotion
        return analyze_emotion

    def test_neutral_text(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("The weather is nice today.")
        self.assertAlmostEqual(score, 0.5)

    def test_emotional_text(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("This is amazing! I love it!")
        self.assertGreater(score, 0.5)

    def test_score_capped_at_max(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("amazing terrible love hate excited sad happy angry wonderful awful !")
        self.assertLessEqual(score, 0.9)

    def test_score_has_minimum(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("ok")
        self.assertGreaterEqual(score, 0.3)


if __name__ == "__main__":
    unittest.main()
