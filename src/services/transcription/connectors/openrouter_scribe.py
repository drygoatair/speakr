"""
OpenRouter transcription connector (ElevenLabs Scribe v2).

Talks to OpenRouter's POST /api/v1/audio/transcriptions endpoint, which takes
base64-encoded audio inside a JSON body (not multipart like OpenAI).
Defaults to the `elevenlabs/scribe-v2` model.
"""

import base64
import logging
import os
import httpx
from typing import Dict, Any, Set, List, Optional

from ..base import (
    BaseTranscriptionConnector,
    TranscriptionCapability,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    ConnectorSpecifications,
)
from ..exceptions import TranscriptionError, ConfigurationError, ProviderError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
DEFAULT_MODEL = 'elevenlabs/scribe-v2'

# MIME type -> OpenRouter `input_audio.format`, used when the filename has no
# usable extension.
_MIME_TO_FORMAT = {
    'audio/mpeg': 'mp3', 'audio/mp3': 'mp3',
    'audio/wav': 'wav', 'audio/x-wav': 'wav', 'audio/wave': 'wav',
    'audio/flac': 'flac', 'audio/x-flac': 'flac',
    'audio/mp4': 'm4a', 'audio/x-m4a': 'm4a', 'audio/m4a': 'm4a',
    'audio/ogg': 'ogg', 'audio/webm': 'webm', 'audio/aac': 'aac',
}

# Words closer together than this (seconds) with the same speaker stay in one
# segment; a speaker change always starts a new one.
_MAX_SEGMENT_GAP = 2.0


class OpenRouterScribeConnector(BaseTranscriptionConnector):
    """Connector for ElevenLabs Scribe v2 through OpenRouter."""

    CAPABILITIES: Set[TranscriptionCapability] = {
        TranscriptionCapability.DIARIZATION,
        TranscriptionCapability.TIMESTAMPS,
        TranscriptionCapability.LANGUAGE_DETECTION,
        TranscriptionCapability.HOTWORDS,
    }
    PROVIDER_NAME = "openrouter_scribe"

    # OpenRouter rejects payloads over 25MB (HTTP 413). Audio is sent base64
    # encoded (+33%), so keep the raw file comfortably below 25MB / 1.33.
    SPECIFICATIONS = ConnectorSpecifications(
        max_file_size_bytes=18 * 1024 * 1024,
        handles_chunking_internally=False,
        recommended_chunk_seconds=600,
        unsupported_codecs=frozenset({'opus'}),
    )

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict with keys:
                - api_key: OpenRouter API key (required)
                - base_url: API base URL (default: https://openrouter.ai/api/v1)
                - model: Model slug (default: elevenlabs/scribe-v2)
                - timeout: Request timeout in seconds (default: 1800)
                - zdr: Only route to zero-data-retention providers (default: True)
        """
        super().__init__(config)
        self.api_key = config['api_key']
        self.base_url = (config.get('base_url') or DEFAULT_BASE_URL).rstrip('/')
        self.model = config.get('model') or DEFAULT_MODEL
        self.timeout = float(config.get('timeout') or 1800)
        self.zdr = bool(config.get('zdr', True))

    def _validate_config(self) -> None:
        if not self.config.get('api_key'):
            raise ConfigurationError("api_key is required for OpenRouter Scribe connector")

    def _headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/murtaza-nasir/speakr',
            'X-Title': 'Speakr - AI Audio Transcription',
        }

    @staticmethod
    def _audio_format(request: TranscriptionRequest) -> str:
        ext = os.path.splitext(request.filename or '')[1].lstrip('.').lower()
        if ext:
            return ext
        mime = (request.mime_type or '').split(';')[0].strip().lower()
        return _MIME_TO_FORMAT.get(mime, 'mp3')

    def _build_payload(self, request: TranscriptionRequest, model: str) -> Dict[str, Any]:
        audio_bytes = request.audio_file.read()
        payload: Dict[str, Any] = {
            'model': model,
            'input_audio': {
                'data': base64.b64encode(audio_bytes).decode('ascii'),
                'format': self._audio_format(request),
            },
            # verbose_json is required for diarization, segments and timestamps
            'response_format': 'verbose_json',
            'timestamp_granularities': ['word', 'segment'],
        }
        if request.language:
            payload['language'] = request.language
        if request.temperature is not None:
            payload['temperature'] = request.temperature
        if request.diarize:
            payload['diarize'] = True
        if request.hotwords:
            keyterms = [k.strip() for k in request.hotwords.split(',') if k.strip()]
            if keyterms:
                payload['keyterms'] = keyterms
        if self.zdr:
            payload['provider'] = {'zdr': True}
        return payload

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        model = self._effective_model(request)
        payload = self._build_payload(request, model)
        url = f'{self.base_url}/audio/transcriptions'

        logger.info(f"Sending request to OpenRouter with model: {model} (diarize={request.diarize})")
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=60.0)) as client:
                response = client.post(url, headers=self._headers(), json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            detail = self._error_detail(e.response)
            status = e.response.status_code
            logger.error(f"OpenRouter request failed ({status}): {detail}")
            raise ProviderError(
                f"OpenRouter request failed with status {status}: {detail}",
                provider=self.PROVIDER_NAME,
                status_code=status,
            ) from e
        except Exception as e:
            logger.error(f"OpenRouter transcription failed: {e}")
            raise TranscriptionError(f"OpenRouter transcription failed: {e}") from e

        return self._parse_response(data, model, diarized=request.diarize)

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            err = body.get('error', body)
            if isinstance(err, dict):
                return str(err.get('message') or err)
            return str(err)
        except Exception:
            return (response.text or '')[:500]

    def _parse_response(self, data: Dict[str, Any], model: str,
                        diarized: bool = False) -> TranscriptionResponse:
        text = (data.get('text') or '').strip()
        words = [w for w in (data.get('words') or []) if isinstance(w, dict)]
        raw_segments = [s for s in (data.get('segments') or []) if isinstance(s, dict)]

        segments: List[TranscriptionSegment] = []
        if diarized and any(w.get('speaker') is not None or w.get('speaker_id') is not None
                            for w in words):
            segments = self._segments_from_words(words)
        elif raw_segments:
            for s in raw_segments:
                seg_text = (s.get('text') or '').strip()
                if not seg_text:
                    continue
                speaker = s.get('speaker', s.get('speaker_id'))
                segments.append(TranscriptionSegment(
                    text=seg_text,
                    speaker=self._label(speaker) if speaker is not None else None,
                    start_time=s.get('start'),
                    end_time=s.get('end'),
                ))

        speakers = sorted({s.speaker for s in segments if s.speaker})
        if not text and segments:
            text = ' '.join(s.text for s in segments)

        return TranscriptionResponse(
            text=text,
            segments=segments or None,
            language=data.get('language'),
            duration=data.get('duration'),
            speakers=speakers or None,
            provider=self.PROVIDER_NAME,
            model=model,
            raw_response=data,
        )

    @staticmethod
    def _label(speaker: Any) -> str:
        """Normalise provider speaker ids (`speaker_0`, 0, 'A') to SPEAKER_NN."""
        s = str(speaker)
        digits = s.rsplit('_', 1)[-1]
        if digits.isdigit():
            return f"SPEAKER_{int(digits):02d}"
        return s if s.upper().startswith('SPEAKER') else f"SPEAKER_{s}"

    def _segments_from_words(self, words: List[Dict[str, Any]]) -> List[TranscriptionSegment]:
        """Group consecutive same-speaker words into segments.

        Scribe emits spacing as separate `type: "spacing"` entries; those are
        folded into the text rather than treated as words.
        """
        segments: List[TranscriptionSegment] = []
        cur_speaker: Optional[str] = None
        cur_text = ''
        cur_start: Optional[float] = None
        cur_end: Optional[float] = None
        cur_words: List[Dict[str, Any]] = []

        def flush():
            if cur_text.strip():
                segments.append(TranscriptionSegment(
                    text=cur_text.strip(), speaker=cur_speaker,
                    start_time=cur_start, end_time=cur_end,
                    words=cur_words or None,
                ))

        for w in words:
            wtext = w.get('text', w.get('word', '')) or ''
            if w.get('type') == 'spacing':
                cur_text += wtext
                continue
            raw_speaker = w.get('speaker', w.get('speaker_id'))
            speaker = self._label(raw_speaker) if raw_speaker is not None else cur_speaker
            start, end = w.get('start'), w.get('end')
            gap_break = (cur_end is not None and start is not None
                         and start - cur_end > _MAX_SEGMENT_GAP)
            if cur_text and (speaker != cur_speaker or gap_break):
                flush()
                cur_text, cur_start, cur_end, cur_words = '', None, None, []
            if not cur_text:
                cur_start = start
            if cur_text and not cur_text.endswith((' ', '\n')) and not wtext.startswith(' '):
                cur_text += ' '
            cur_text += wtext
            cur_end = end if end is not None else cur_end
            cur_speaker = speaker
            cur_words.append({'word': wtext.strip(), 'start': start, 'end': end})
        flush()
        return segments

    def list_models(self) -> List[Dict[str, Any]]:
        """Return OpenRouter models that accept audio input for transcription."""
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(f'{self.base_url}/models', headers=self._headers())
                resp.raise_for_status()
                models = resp.json().get('data', [])
        except Exception as e:
            logger.warning(f"openrouter /models probe failed: {e}")
            return []
        keep = ('scribe', 'whisper', 'transcribe', 'voxtral')
        return [
            {'id': m['id'], 'label': m.get('name') or m['id'], 'owned_by': m['id'].split('/')[0]}
            for m in models
            if m.get('id') and any(k in m['id'].lower() for k in keep)
        ]

    def health_check(self) -> bool:
        """Verify the API key against OpenRouter's key endpoint."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f'{self.base_url}/key', headers=self._headers())
                return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["api_key"],
            "properties": {
                "api_key": {"type": "string", "description": "OpenRouter API key"},
                "base_url": {"type": "string", "default": DEFAULT_BASE_URL,
                             "description": "OpenRouter API base URL"},
                "model": {"type": "string", "default": DEFAULT_MODEL,
                          "description": "OpenRouter STT model slug"},
                "timeout": {"type": "integer", "default": 1800,
                            "description": "Request timeout in seconds"},
                "zdr": {"type": "boolean", "default": False,
                        "description": "Restrict routing to zero-data-retention providers"},
            }
        }
