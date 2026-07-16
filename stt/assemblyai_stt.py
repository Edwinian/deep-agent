"""AssemblyAI speech-to-text client for prerecorded audio."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import assemblyai as aai
from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """A speaker-labeled transcript segment."""

    speaker: str
    text: str
    start: int | None = None
    end: int | None = None


class TranscriptionResult(BaseModel):
    """Normalized prerecorded transcription response."""

    text: str
    utterances: list[Utterance] = Field(default_factory=list)
    language_code: str | None = None
    audio_duration: float | None = None


class AssemblyAISTT:
    """AssemblyAI client for prerecorded speech-to-text (no WebSocket)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY is required. Set it in the environment or pass api_key."
            )
        aai.settings.api_key = self.api_key

    def transcribe(
        self,
        audio: str | Path | BinaryIO | bytes,
        *,
        speech_models: list[str] | None = None,
        language_detection: bool = True,
        speaker_labels: bool = True,
    ) -> TranscriptionResult:
        """Transcribe prerecorded audio via AssemblyAI.

        Args:
            audio: Local file path, remote URL, file-like object, or raw bytes.
            speech_models: Preferred AssemblyAI speech models to try in order.
            language_detection: Detect spoken language automatically.
            speaker_labels: Enable speaker diarization.

        Returns:
            Normalized transcription text and optional speaker utterances.

        Raises:
            RuntimeError: If AssemblyAI returns an error status.
        """
        if isinstance(audio, Path):
            data: str | BinaryIO = str(audio)
        elif isinstance(audio, bytes):
            data = BytesIO(audio)
        else:
            data = audio

        config = aai.TranscriptionConfig(
            speech_models=speech_models
            or ["universal-3-5-pro", "universal-2"],
            language_detection=language_detection,
            speaker_labels=speaker_labels,
        )
        try:
            transcript = aai.Transcriber().transcribe(data, config=config)
        except aai.types.TranscriptError as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        utterances = [
            Utterance(
                speaker=str(utterance.speaker),
                text=str(utterance.text),
                start=getattr(utterance, "start", None),
                end=getattr(utterance, "end", None),
            )
            for utterance in (transcript.utterances or [])
        ]

        return TranscriptionResult(
            text=str(transcript.text or ""),
            utterances=utterances,
            language_code=getattr(transcript, "language_code", None),
            audio_duration=getattr(transcript, "audio_duration", None),
        )
