"""CPU queue proxy; native model classes never cross the engine boundary."""
from types import SimpleNamespace

_runtime = None


def runtime_for_jobs():
    if _runtime is None:
        raise RuntimeError('Managed runtime has not been attached')
    return _runtime


class ModelRunner:
    def __init__(self, model_path, *, device='cuda:0', dtype='fp32'):
        if device != 'cuda:0' or dtype != 'fp32':
            raise ValueError('Native CUDA/FP32 configuration must be retained')
        self.model_path, self.device_name, self.dtype_name = str(model_path), device, dtype

    def runtime_info(self):
        return dict(backend='hf', path=self.model_path, device=self.device_name, dtype=self.dtype_name)

    @property
    def is_loaded(self):
        return runtime_for_jobs().status()['residency'] == 'loaded'

    def transcribe(self, audio_path, *, status_callback=None, **options):
        runtime = runtime_for_jobs()
        if status_callback:
            status_callback('loading_model', 0.05, None)
        with runtime.execution():
            if status_callback:
                status_callback('transcribing', 0.1, None)
            data = runtime.get().transcribe(str(audio_path), options)
        if status_callback:
            status_callback('transcribing', 0.85, data['generated_tokens'])
        return SimpleNamespace(**data)
