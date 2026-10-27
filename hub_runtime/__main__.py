import os


def create_app():
    from ttd_model_runtime import Runtime
    from ttd_model_runtime.integrations.fastapi import attach
    from . import runner
    from .contract import DEFAULT_PROMPT
    from .engine import load_model, completion, cleanup
    from moss_transcribe_diarize.app.server import create_app as native_app
    runtime = Runtime(load_model, completion=completion, cleanup=cleanup, execution_timeout=None)
    runner._runtime = runtime
    # Distinct ownership prevents startup recovery from touching Backup jobs.
    os.environ['MTD_WORKER_ID'] = os.environ['HUB_RUNTIME_ACTOR_ID']
    app = native_app(model_path='/models/MOSS-Transcribe-Diarize', runs_dir='/data/runs',
        device='cuda:0', dtype='fp32', prompt=DEFAULT_PROMPT, max_length=131072,
        max_new_tokens=8192, decoding='greedy', temperature=None, backend='hf')
    runtime.pending_work = lambda: app.state.manager._queue.unfinished_tasks
    return attach(app, runtime=runtime)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host='0.0.0.0', port=8000, workers=1)
