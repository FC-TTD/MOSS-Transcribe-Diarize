import contextvars
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

from hub_runtime import runner
from hub_runtime.contract import DEFAULT_PROMPT
from moss_transcribe_diarize.app.jobs import JobManager
from moss_transcribe_diarize.app.server import create_app

trace = contextvars.ContextVar('acceptance_trace', default=None)


class FakeRuntime:
    def __init__(self):
        self.accepted, self.completed = [], []
    def accept_background_work(self): self.accepted.append(trace.get())
    def complete_background_work(self): self.completed.append(trace.get())


class FakeRunner:
    model_path = '/models/MOSS-Transcribe-Diarize'
    def __init__(self, unknown=False): self.unknown = unknown; self.traces = []
    def transcribe(self, *args, **kwargs):
        self.traces.append(trace.get())
        if self.unknown:
            error = RuntimeError('Native outcome unknown')
            error.code = 'execution_unknown'
            raise error
        # End known failure without subtitle/ffmpeg to test queue accounting.
        raise ValueError('Known invalid media')


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.runtime = FakeRuntime(); runner._runtime = self.runtime
        os.environ['MTD_WORKER_ID'] = 'new-managed-actor'
    def manager(self, native):
        return JobManager(self.path, native, prompt=DEFAULT_PROMPT, max_length=131072, max_new_tokens=8192)
    def wait(self, manager):
        deadline = time.monotonic() + 5
        while manager._queue.unfinished_tasks and time.monotonic() < deadline: time.sleep(.01)
        self.assertEqual(manager._queue.unfinished_tasks, 0)
    def test_parent_import_and_contract(self):
        app = create_app(model_path='/models/MOSS-Transcribe-Diarize', runs_dir=self.path, device='cuda:0', dtype='fp32', max_new_tokens=8192)
        schema = app.openapi()
        self.assertIn('/api/jobs/{job_id}/rerun', schema['paths'])
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(app.state.manager.max_new_tokens, 8192)
        self.assertIn('fp32', str(app.state.manager.model_runner.runtime_info()))
    def test_serial_queue_retains_request_context(self):
        native = FakeRunner(); manager = self.manager(native)
        trace.set('first-scoped-acceptance')
        job, source = manager.create_job_for_upload('sample.wav'); source.write_bytes(b'audio')
        manager.enqueue(job.id); self.wait(manager)
        self.assertEqual(native.traces, ['first-scoped-acceptance'])
        self.assertEqual(self.runtime.accepted, self.runtime.completed)
        self.assertEqual(self.runtime.completed, ['first-scoped-acceptance'])
    def test_unknown_preserves_hold_and_blocks_replay_delete(self):
        native = FakeRunner(unknown=True); manager = self.manager(native)
        job, source = manager.create_job_for_upload('sample.wav'); source.write_bytes(b'audio')
        manager.enqueue(job.id); self.wait(manager)
        self.assertEqual(manager.get_job(job.id).status, 'unknown')
        self.assertEqual(self.runtime.completed, [])
        with self.assertRaises(RuntimeError): manager.rerun_job(job.id)
        with self.assertRaises(RuntimeError): manager.delete_job(job.id)
        self.assertTrue(source.exists())
    def test_startup_preserves_backup_jobs(self):
        folder = self.path/'old'; folder.mkdir()
        state = dict(id='old', worker_id='backup-native', status='transcribing', input_path=str(folder/'input.wav'), job_dir=str(folder))
        path = folder/'job.json'; path.write_text(json.dumps(state)); before = path.read_bytes()
        manager = self.manager(FakeRunner())
        self.assertEqual(manager.get_job('old').status, 'transcribing')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.runtime.accepted, [])
    def test_native_device_and_precision_guard(self):
        for device, dtype in [('cpu','fp32'), ('cuda:0','bf16')]:
            with self.assertRaises(ValueError): runner.ModelRunner('/models/MOSS-Transcribe-Diarize',device=device,dtype=dtype)


if __name__ == '__main__': unittest.main()
