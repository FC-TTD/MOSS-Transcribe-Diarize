"""Original HF model, loaded only inside the SDK-assigned GPU process."""
import os


class Engine:
    def __init__(self):
        import torch
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible.startswith('GPU-') or ',' in visible or not torch.cuda.is_available():
            raise RuntimeError('An assigned, available CUDA UUID is required')
        from moss_transcribe_diarize.app.model_runner import ModelRunner
        self.runner = ModelRunner('/models/MOSS-Transcribe-Diarize', device='cuda:0', dtype='fp32')
        self.runner._ensure_loaded()
        assert self.runner._device.type == 'cuda' and self.runner._dtype == torch.float32

    def transcribe(self, audio_path, options):
        # No callback closure, model instance or native dataclass is unpickled
        # in the HTTP parent; the native output text and timing are unchanged.
        return self.runner.transcribe(audio_path, **options).to_dict()


def load_model():
    return Engine()


def completion(model):
    import torch
    torch.cuda.synchronize()
    return True


def cleanup():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
