"""
Microsoft MAI-Transcribe 2 connector (via OpenRouter).

Calls OpenRouter's speech-to-text endpoint (POST /api/v1/audio/transcriptions)
with model ``microsoft/mai-transcribe-2``. Audio is sent inline as base64 in a
JSON body. ``verbose_json`` is requested so segment timestamps (and speaker
labels) come back. Diarization, keyword biasing and transcription style are
passed through ``provider.options.azure`` (Microsoft's MAI-Transcribe options).
Language is auto-detected, including code switching, unless pinned.
"""

import base64
import logging
import os
from typing import Any, Dict, Optional, Set

import httpx

from ..base import (
    BaseTranscriptionConnector,
    ConnectorSpecifications,
    TranscriptionCapability,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
)
from ..exceptions import ConfigurationError, ProviderError, TranscriptionError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "microsoft/mai-transcribe-2"

# Map MIME types / extensions to the `input_audio.format` value.
_FORMAT_ALIASES = {
    "mpeg": "mp3", "mpga": "mp3", "x-wav": "wav", "wave": "wav",
    "x-m4a": "m4a", "mp4": "m4a", "x-flac": "flac", "oga": "ogg",
}


def _audio_format(filename: str, mime_type: Optional[str]) -> str:
    ext = os.path.splitext(filename or "")[1].lstrip(".").lower()
    if not ext and mime_type and "/" in mime_type:
        ext = mime_type.split("/", 1)[1].split(";")[0].strip().lower()
    return _FORMAT_ALIASES.get(ext, ext) or "wav"


class OpenRouterMAITranscribeConnector(BaseTranscriptionConnector):
    """Connector for microsoft/mai-transcribe-2 through OpenRouter."""

    CAPABILITIES: Set[TranscriptionCapability] = {
        TranscriptionCapability.CHUNKING,
        TranscriptionCapability.TIMESTAMPS,
        TranscriptionCapability.LANGUAGE_DETECTION,
        TranscriptionCapability.DIARIZATION,
        TranscriptionCapability.HOTWORDS,
    }
    PROVIDER_NAME = "openrouter_mai_transcribe"

    # OpenRouter documents a 25 MB limit for file uploads; the model page does
    # not publish its own limits, so stay conservative and let Speakr chunk.
    SPECIFICATIONS = ConnectorSpecifications(
        max_file_size_bytes=25 * 1024 * 1024,
        handles_chunking_internally=False,
        recommended_chunk_seconds=600,
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config["api_key"]
        self.base_url = (config.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.model = config.get("model") or DEFAULT_MODEL
        self.timeout = float(config.get("timeout") or 1800)
        self.verbose_json = bool(config.get("verbose_json", True))
        # "clean" drops fillers/false starts; "verbatim" (API default) keeps them
        self.transcribe_style = (config.get("transcribe_style") or "").strip().lower() or None
        if self.transcribe_style not in (None, "clean", "verbatim"):
            raise ConfigurationError("transcribe_style must be 'clean' or 'verbatim'")
        self.word_timestamps = bool(config.get("word_timestamps", False))

    def _validate_config(self) -> None:
        if not self.config.get("api_key"):
            raise ConfigurationError(
                "api_key (OpenRouter API key) is required for the MAI-Transcribe connector"
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/murtaza-nasir/speakr",
            "X-OpenRouter-Title": "Speakr - AI Audio Transcription",
        }

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        model = self._effective_model(request)
        try:
            audio_bytes = request.audio_file.read()
            body: Dict[str, Any] = {
                "model": model,
                "input_audio": {
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                    "format": _audio_format(request.filename, request.mime_type),
                },
            }
            if request.language:
                body["language"] = request.language
            if request.temperature is not None:
                body["temperature"] = request.temperature
            if self.verbose_json:
                body["response_format"] = "verbose_json"
                if self.word_timestamps:
                    body["timestamp_granularities"] = ["word"]
            azure = self._azure_options(request)
            if azure:
                body["provider"] = {"options": {"azure": azure}}

            logger.info(f"Sending request to OpenRouter STT with model: {model}")
            timeout = httpx.Timeout(self.timeout, connect=60.0)
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._headers(),
                    json=body,
                )
            if resp.status_code >= 400:
                raise ProviderError(
                    f"OpenRouter STT error {resp.status_code}: {self._error_message(resp)}",
                    provider=self.PROVIDER_NAME,
                    status_code=resp.status_code,
                )
            return self._parse_response(resp.json(), model)

        except TranscriptionError:
            raise
        except Exception as e:
            logger.error(f"MAI-Transcribe transcription failed: {e}")
            raise TranscriptionError(f"MAI-Transcribe transcription failed: {e}") from e

    def _azure_options(self, request: TranscriptionRequest) -> Dict[str, Any]:
        azure: Dict[str, Any] = {}
        if request.diarize:
            azure["diarization"] = {"enabled": True}
        phrases = [h.strip() for h in (request.hotwords or "").split(",") if h.strip()]
        if phrases:
            azure["phraseList"] = {"phrases": phrases}
        if self.transcribe_style:
            azure["enhancedMode"] = {"modelOptions": {"transcribeStyle": self.transcribe_style}}
        return azure

    @staticmethod
    def _speaker_label(raw: Any) -> Optional[str]:
        """Normalise provider speaker ids (1, "1", "Guest-1") to Speakr's SPEAKER_NN."""
        if raw is None:
            return None
        try:
            return f"SPEAKER_{int(raw):02d}"
        except (TypeError, ValueError):
            return str(raw)

    @staticmethod
    def _error_message(resp: httpx.Response) -> str:
        try:
            err = resp.json().get("error", {})
            return err.get("message") or resp.text[:500]
        except Exception:
            return resp.text[:500]

    def _parse_response(self, data: Dict[str, Any], model: str) -> TranscriptionResponse:
        segments = None
        speakers = set()
        raw_segments = data.get("segments") or []
        if raw_segments:
            segments = []
            for seg in raw_segments:
                speaker = self._speaker_label(seg.get("speaker"))
                if speaker is None:
                    speaker = seg.get("speaker_label")
                if speaker is not None:
                    speakers.add(speaker)
                segments.append(TranscriptionSegment(
                    text=(seg.get("text") or "").strip(),
                    speaker=speaker,
                    start_time=seg.get("start"),
                    end_time=seg.get("end"),
                    confidence=seg.get("confidence"),
                ))

        usage = data.get("usage") or {}
        return TranscriptionResponse(
            text=data.get("text", ""),
            segments=segments,
            language=data.get("language"),
            duration=data.get("duration") or usage.get("seconds"),
            speakers=sorted(speakers) if speakers else None,
            provider=self.PROVIDER_NAME,
            model=model,
            raw_response=data,
        )

    def health_check(self) -> bool:
        """Verify the API key against OpenRouter's key endpoint."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.base_url}/key",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            return resp.status_code < 400
        except Exception:
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["api_key"],
            "properties": {
                "api_key": {"type": "string", "description": "OpenRouter API key"},
                "base_url": {
                    "type": "string",
                    "default": DEFAULT_BASE_URL,
                    "description": "OpenRouter API base URL",
                },
                "model": {
                    "type": "string",
                    "default": DEFAULT_MODEL,
                    "description": "OpenRouter STT model ID",
                },
                "timeout": {
                    "type": "integer",
                    "default": 1800,
                    "description": "Request timeout in seconds",
                },
                "verbose_json": {
                    "type": "boolean",
                    "default": True,
                    "description": "Request verbose_json (segment timestamps and speaker labels)",
                },
                "word_timestamps": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also request word-level timestamps (timestamp_granularities=[word])",
                },
                "transcribe_style": {
                    "type": "string",
                    "enum": ["clean", "verbatim"],
                    "description": "clean drops fillers and false starts; verbatim (default) keeps them",
                },
            },
        }
