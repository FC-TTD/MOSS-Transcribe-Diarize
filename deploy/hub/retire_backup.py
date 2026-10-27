"""Drain the approved independent Backup actor before its friendly entry moves."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

NAME = 'moss-transcribe-diarize-backup'
EXPECTED = '237e0d890e8fa29028e6fa0952c461fecaaa4e026ee396646d6aac623b402942'
IMAGE = 'registry.ttd/moss-transcribe-diarize@sha256:c60ff82a1be75afe4948511039974a6bbb255d10aceb3b0bcbac00cbd68f19e5'


def inspect():
    return json.loads(subprocess.check_output(['docker', 'inspect', NAME]))[0]


def save(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2)


def main():
    root = Path('/opt/music-hub-release-20261027/asr-retirement')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    original = inspect()
    assert original['Id'] == EXPECTED and original['Config']['Image'] == IMAGE and original['State']['Running']
    assert set(original['NetworkSettings']['Networks']) == {'caddy'}
    assert not (root / 'before.json').exists(), 'Inspect an existing retirement instead of repeating it'
    save(root / 'before.json', original)
    runs = Path(next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/data/runs'))
    history = {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in runs.glob('*/job.json')}
    save(root / 'history-before.json', history)
    # Remove only this old actor's ingress interface. Local API/CPU queue and
    # native GPU work keep running; new requests cannot race the idle probe.
    # The public host key/trust and all shared networks remain unchanged.
    subprocess.run(['docker', 'network', 'disconnect', 'caddy', NAME], check=True)
    code = ("import json,urllib.request;d=json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/jobs'));"
            "a=[dict(id=x['id'],status=x['status']) for x in d['jobs'] if x['status'] in "
            "['queued','loading_model','transcribing','postprocessing','rendering']];print(json.dumps(a))")
    deadline = time.monotonic() + 1800
    empty = 0
    observations = []
    while empty < 3:
        active = json.loads(subprocess.check_output(['docker', 'exec', NAME, 'python', '-c', code]))
        observations.append(dict(at=time.time(), active=active))
        empty = empty + 1 if not active else 0
        if time.monotonic() > deadline:
            raise RuntimeError('Backup still has native work; preserve the ingress fence and inspect, never force kill')
        time.sleep(2)
    subprocess.run(['docker', 'stop', '--timeout', '-1', NAME], check=True, stdout=subprocess.DEVNULL)
    stopped = inspect()
    assert not stopped['State']['Running'] and not stopped['State']['OOMKilled']
    changed = [identity for identity, digest in history.items()
        if not (runs / identity / 'job.json').is_file() or hashlib.sha256((runs / identity / 'job.json').read_bytes()).hexdigest() != digest]
    receipt = dict(stopped=True, active_jobs=0, container=EXPECTED, image=IMAGE,
        exit_code=stopped['State']['ExitCode'], observations=observations, original_history=len(history),
        history_changed_during_drain=changed, volumes_preserved=True, ingress_network_disconnected=True)
    save(root / 'backup-retired.json', receipt)
    # An application rollback first closes/drains the new managed ASR actor.
    # Then reconnect/start this preserved old container with its recorded IP and
    # aliases; never restore the run history over new user edits.
    print(json.dumps({k: v for k, v in receipt.items() if k != 'observations'}))


if __name__ == '__main__':
    main()
