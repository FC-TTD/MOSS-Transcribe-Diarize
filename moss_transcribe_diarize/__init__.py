"""Public native exports, loaded only when their particular capability is used."""
from importlib import import_module
_EXPORTS = {'MossTranscribeDiarizeConfig': ('configuration_moss_transcribe_diarize', 'MossTranscribeDiarizeConfig'), 'MossTranscribeDiarizeForConditionalGeneration': ('modeling_moss_transcribe_diarize', 'MossTranscribeDiarizeForConditionalGeneration'), 'MossTranscribeDiarizeModel': ('modeling_moss_transcribe_diarize', 'MossTranscribeDiarizeModel'), 'MossTranscribeDiarizePreTrainedModel': ('modeling_moss_transcribe_diarize', 'MossTranscribeDiarizePreTrainedModel'), 'VQAdaptor': ('modeling_moss_transcribe_diarize', 'VQAdaptor'), 'MossTranscribeDiarizeProcessor': ('processing_moss_transcribe_diarize', 'MossTranscribeDiarizeProcessor'), 'SubtitleSegment': ('subtitle', 'SubtitleSegment'), 'SubtitleStyle': ('subtitle', 'SubtitleStyle'), 'coerce_subtitle_segments': ('subtitle', 'coerce_subtitle_segments'), 'export_ass': ('subtitle', 'export_ass'), 'export_json': ('subtitle', 'export_json'), 'export_srt': ('subtitle', 'export_srt'), 'normalize_segments': ('subtitle', 'normalize_segments'), 'subtitle_segments_from_transcript': ('subtitle', 'subtitle_segments_from_transcript'), 'TranscriptParseError': ('transcript_parser', 'TranscriptParseError'), 'TranscriptSegment': ('transcript_parser', 'TranscriptSegment'), 'TranscriptStreamParser': ('transcript_parser', 'TranscriptStreamParser'), 'iter_transcript_segments': ('transcript_parser', 'iter_transcript_segments'), 'parse_transcript': ('transcript_parser', 'parse_transcript')}
__all__ = list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), attribute)
    globals()[name] = value
    return value
