import os

# HuggingFace caches the Kokoro snapshot in ~/.cache/huggingface. Once it's
# present, skip the per-launch revision check (the "Fetching 63 files" scan)
# by forcing offline mode. Set LOCAL_COACH_HF_ONLINE=1 to override (e.g. to
# pull a newer model revision).
if os.environ.get("LOCAL_COACH_HF_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Suppress loky's "leaked semaphore" warning at shutdown. We use os._exit on
# Ctrl+C to avoid an MLX/Metal teardown crash, which skips loky's atexit
# cleanup. The warning is cosmetic — the OS reclaims the semaphore anyway.
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be \d+ leaked semaphore",
)

import time
import threading
import re
import subprocess
import numpy as np
import whisper
import sounddevice as sd
import argparse
from queue import Queue
from rich.console import Console
# Updated imports for modern LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from tts import KokoroTTS

console = Console()
stt = whisper.load_model("tiny.en")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Local Voice Assistant (Kokoro TTS)")
parser.add_argument("--model", type=str, default=None, help="Ollama model name (default: coach)")
parser.add_argument("--provider", type=str, default="ollama", choices=["ollama"],
                    help="LLM provider: only 'ollama' (local). Cloud providers removed to keep chat data local.")
parser.add_argument("--no-tts", action="store_true", help="Disable TTS (alias for --tts none)")
parser.add_argument("--tts", type=str, default=None, choices=["none", "kokoro", "say"],
                    help="TTS backend: 'none' (text only), 'kokoro' (MLX, default), 'say' (macOS built-in)")
parser.add_argument("--kokoro-voice", type=str,
                    default=os.environ.get("LOCAL_COACH_KOKORO_VOICE", "bf_emma"),
                    help="Kokoro voice id (e.g. bf_emma, af_heart, am_adam). "
                         "Defaults to $LOCAL_COACH_KOKORO_VOICE or 'bf_emma'.")
parser.add_argument("--kokoro-speed", type=float, default=1.0, help="Kokoro speed multiplier")
parser.add_argument("--say-voice", type=str, default=None, help="macOS 'say' voice name (e.g. 'Samantha', 'Daniel')")
parser.add_argument("--say-rate", type=int, default=None, help="macOS 'say' words-per-minute rate (default ~175)")
parser.add_argument("--debug", action="store_true", help="Print timing info for each pipeline stage")
args = parser.parse_args()

# Resolve TTS mode: explicit --tts wins; otherwise --no-tts → 'none'; default kokoro.
if args.tts is not None:
    tts_mode = args.tts
elif args.no_tts:
    tts_mode = "none"
else:
    tts_mode = "kokoro"

def dlog(msg: str) -> None:
    """Debug log with monotonic timestamp."""
    if args.debug:
        console.print(f"[dim][{time.monotonic():8.2f}s] {msg}[/dim]")

# Load Kokoro only if it's the selected backend (model load is non-trivial).
tts = (
    KokoroTTS(default_voice=args.kokoro_voice, speed=args.kokoro_speed)
    if tts_mode == "kokoro"
    else None
)


def create_llm(provider: str, model: str | None = None):
    """
    Create a local LLM instance.

    Args:
        provider: LLM provider name. Only 'ollama' is supported — cloud providers
            were removed so chat data cannot leave the machine.
        model: Ollama model name. Defaults to 'coach'.

    Returns:
        A LangChain LLM instance backed by a local Ollama server.
    """
    if provider == "ollama":
        return OllamaLLM(model=model or "coach", base_url="http://localhost:11434")
    raise ValueError(f"Unknown provider: {provider}. Supported: ollama")


# System prompt is owned by the Modelfile (see ./Modelfile). Don't duplicate it here.
prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM via provider factory
llm = create_llm(
    provider=args.provider,
    model=args.model,
)

chain = prompt_template | llm | StrOutputParser()

# In-memory conversation history (list of HumanMessage / AIMessage)
chat_history: list = []

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    t0 = time.monotonic()
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    dlog(f"whisper transcribe ({audio_np.size / 16000:.2f}s audio) in {time.monotonic() - t0:.2f}s")
    return text


def get_llm_response(text: str) -> str:
    chat_history.append(HumanMessage(content=text))
    response = chain.invoke({"history": chat_history[:-1], "input": text})
    result = response.strip()
    chat_history.append(AIMessage(content=result))
    return result


_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")

# Markdown decorations Kokoro would otherwise read aloud (e.g. "asterisk").
_MD_EMPHASIS = re.compile(r"(\*\*|\*|__|_|`)(.+?)\1")
_MD_STRAY = re.compile(r"[*_`#>]+")


def strip_markdown_for_tts(text: str) -> str:
    """Remove markdown decorations so Kokoro doesn't pronounce them literally."""
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_STRAY.sub("", text)
    return text


def stream_sentences(text_iter):
    """
    Consume an iterator of text chunks and yield complete sentences as soon as
    a terminal punctuation boundary is seen. Trailing partial text is yielded last.
    """
    buf = ""
    for chunk in text_iter:
        if not chunk:
            continue
        buf += chunk
        while True:
            m = _SENTENCE_END.search(buf)
            if not m:
                break
            end = m.end()
            sentence = buf[:end].strip()
            buf = buf[end:]
            if sentence:
                yield sentence
    tail = buf.strip()
    if tail:
        yield tail


def stream_llm_to_text(text: str) -> str:
    """Stream LLM tokens directly to the console as they arrive. Returns the full text."""
    chat_history.append(HumanMessage(content=text))
    parts: list[str] = []
    console.print("[cyan]Assistant: ", end="")
    dlog("Fetching LLM response...")
    t_start = time.monotonic()
    first_token_logged = False
    for chunk in chain.stream({"history": chat_history[:-1], "input": text}):
        if not chunk:
            continue
        if not first_token_logged:
            dlog(f"LLM first token after {time.monotonic() - t_start:.2f}s")
            first_token_logged = True
        parts.append(chunk)
        console.print(chunk, end="", highlight=False, markup=False)
    console.print("")
    dlog(f"LLM full response in {time.monotonic() - t_start:.2f}s ({len(''.join(parts))} chars)")
    result = "".join(parts).strip()
    chat_history.append(AIMessage(content=result))
    return result


def stream_llm_to_say(text: str, say_voice: str | None, say_rate: int | None) -> str:
    """
    Stream LLM tokens, segment into sentences, and speak each via macOS `say`.
    Sentences queue and play sequentially; LLM keeps generating in the background.
    """
    chat_history.append(HumanMessage(content=text))
    speak_queue: Queue = Queue()
    full_text_parts: list[str] = []
    t_start = time.monotonic()

    def producer():
        try:
            first_token_logged = False
            first_sentence_logged = False
            def chunked():
                nonlocal first_token_logged
                for chunk in chain.stream({"history": chat_history[:-1], "input": text}):
                    if chunk and not first_token_logged:
                        dlog(f"LLM first token after {time.monotonic() - t_start:.2f}s")
                        first_token_logged = True
                    yield chunk
            for sentence in stream_sentences(chunked()):
                if not first_sentence_logged:
                    dlog(f"first sentence ready after {time.monotonic() - t_start:.2f}s: {sentence!r}")
                    first_sentence_logged = True
                full_text_parts.append(sentence + " ")
                speak_queue.put(sentence)
            dlog(f"LLM full response in {time.monotonic() - t_start:.2f}s")
        finally:
            speak_queue.put(None)

    def consumer():
        first_play_logged = False
        while True:
            sentence = speak_queue.get()
            if sentence is None:
                return
            console.print(f"[cyan]Assistant: {sentence}")
            cmd = ["say"]
            if say_voice:
                cmd += ["-v", say_voice]
            if say_rate is not None:
                cmd += ["-r", str(say_rate)]
            cmd += [strip_markdown_for_tts(sentence)]
            t_say = time.monotonic()
            subprocess.run(cmd, check=False)
            if not first_play_logged:
                dlog(f"first audio finished {time.monotonic() - t_start:.2f}s after request "
                     f"(say took {time.monotonic() - t_say:.2f}s)")
                first_play_logged = True

    prod = threading.Thread(target=producer, daemon=True)
    cons = threading.Thread(target=consumer, daemon=True)
    prod.start()
    cons.start()
    prod.join()
    cons.join()

    result = "".join(full_text_parts).strip()
    chat_history.append(AIMessage(content=result))
    return result


def stream_llm_to_kokoro(text: str, voice: str) -> str:
    """
    Stream LLM tokens on a background thread, segment into sentences, and run
    Kokoro synthesis + playback on the main thread. MLX streams are tied to
    the thread that loaded the model, so all `mx.*` calls must happen here.
    """
    chat_history.append(HumanMessage(content=text))
    sentence_queue: Queue = Queue()
    full_text_parts: list[str] = []
    t_start = time.monotonic()

    def llm_worker():
        try:
            dlog("Fetching LLM response...")
            first_token_logged = False
            def chunked():
                nonlocal first_token_logged
                for chunk in chain.stream({"history": chat_history[:-1], "input": text}):
                    if chunk and not first_token_logged:
                        dlog(f"LLM first token after {time.monotonic() - t_start:.2f}s")
                        first_token_logged = True
                    yield chunk
            for sentence in stream_sentences(chunked()):
                full_text_parts.append(sentence + " ")
                sentence_queue.put(sentence)
        finally:
            sentence_queue.put(None)

    worker = threading.Thread(target=llm_worker, daemon=True)
    worker.start()

    first_play_logged = False
    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            break
        spoken = strip_markdown_for_tts(sentence)
        t_tts = time.monotonic()
        sample_rate, audio_array = tts.synthesize(spoken, voice=voice)
        dlog(f"Kokoro sentence ({len(sentence)} chars) in {time.monotonic() - t_tts:.2f}s")
        console.print(f"[cyan]Assistant: {sentence}")
        if not first_play_logged:
            dlog(f"first playback starting at {time.monotonic() - t_start:.2f}s")
            first_play_logged = True
        sd.play(audio_array, sample_rate)
        sd.wait()

    worker.join()
    result = "".join(full_text_parts).strip()
    chat_history.append(AIMessage(content=result))
    return result


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant (Kokoro TTS)")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if tts_mode == "none":
        console.print("[yellow]TTS disabled — assistant replies will stream as text")
    elif tts_mode == "say":
        v = args.say_voice or "system default"
        r = args.say_rate if args.say_rate is not None else "default"
        console.print(f"[green]TTS: macOS 'say' (voice: {v}, rate: {r})")
    else:
        console.print(f"[green]TTS: Kokoro (voice: {args.kokoro_voice}, speed: {args.kokoro_speed})")

    console.print(f"[blue]LLM model: {args.model or 'coach'}")
    console.print(f"[blue]LLM provider: {args.provider}")
    if args.debug:
        console.print("[magenta]Debug timing enabled")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    try:
        while True:
            console.input(
                "🎤 Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                if tts_mode == "none":
                    response = stream_llm_to_text(text)
                elif tts_mode == "say":
                    response = stream_llm_to_say(
                        text,
                        say_voice=args.say_voice,
                        say_rate=args.say_rate,
                    )
                else:
                    response = stream_llm_to_kokoro(text, voice=args.kokoro_voice)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
        os._exit(0)

    console.print("[blue]Session ended.")
