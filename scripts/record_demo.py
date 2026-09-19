"""Capture a bounded, continuous local execution for the static Replay page."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import psutil

from neuroterrarium.data import sha256_file
from neuroterrarium.recording import ExecutionJournal, FrameBuffer, frame
from neuroterrarium.registry import default_path
from neuroterrarium.runtime import BEHAVIOR_SOURCE_FILES, Session, _digest
from neuroterrarium.training import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=800)
    args = parser.parse_args()
    if not 100 <= args.windows <= 2000:
        parser.error('windows must be between 100 and 2000')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    session = Session(args.data_dir, args.models, seed=519, scenario='open')
    frames = FrameBuffer()
    journal = ExecutionJournal(args.output / 'execution', session)
    try:
        # The same placed resource is supplied to each body in this play scene.
        # These are logged environment edits, not inputs hidden in a controller.
        for body in session.world.bodies:
            command = {'type': 'food_add', 'x': body.x, 'y': body.y}
            session.execute(command)
            journal.write({'type': 'command', 'command': command})
        for step in range(args.windows):
            if step % 25 == 0:
                if psutil.virtual_memory().available < 3 * 1024**3:
                    raise RuntimeError('Less than 3 GiB system memory available')
                if shutil.disk_usage(args.output).free < 10 * 1024**3:
                    raise RuntimeError('Less than 10 GiB disk space available')
            if step == 100:
                command = {'type': 'stimulus_add', 'x': 40, 'y': 28,
                           'radius': 1, 'growth': 0.6, 'physical': False}
                session.execute(command)
                journal.write({'type': 'command', 'command': command})
            session.step()
            journal.advanced(session)
            if step % 5 == 0:
                frames.append(frame(session, 0))
        recording = {'schema': 'neuroterrarium.frame-replay.v1', 'mode': 'Replay',
                     'verification': 'recorded event/action playback; no controller recomputation',
                     'frames': frames.export()}
        path = args.output / 'representative-replay.json'
        path.write_text(json.dumps(recording, separators=(',', ':'), allow_nan=False) + '\n')
        source_files = {name: sha256_file(Path(__file__).parents[1] / 'src/neuroterrarium' / name)
                        for name in BEHAVIOR_SOURCE_FILES}
        if _digest(source_files) != session.behavior_sha256:
            raise RuntimeError('Behavior source changed while recording')
        models = json.loads((args.models / 'manifest.json').read_text())
        manifest = {'schema': 'neuroterrarium.replay-manifest.v1', 'sha256': sha256_file(path),
                    'bytes': path.stat().st_size, 'frames': len(recording['frames']),
                    'source_sha256': _digest(source_files), 'source_files': source_files,
                    'data_manifest_sha256': sha256_file(default_path().with_name('data-v783.json')),
                    'interface_sha256': sha256_file(default_path().with_name('brain-interface.json')),
                    'interface_registry_sha256': sha256_file(default_path()),
                    'models_manifest_sha256': sha256_file(args.models / 'manifest.json'),
                    'model_set_sha256': _digest(models)}
        atomic_json(args.output / 'replay-manifest.json', manifest)
        atomic_json(args.output / 'capture.json', {'status': 'completed', 'action_windows': args.windows,
                    'simulation_seconds': session.world.time, 'wall_seconds': time.perf_counter() - started,
                    'purpose': 'Recorded local play with equal resource placement and a visual shadow; not a formal experiment',
                    'recording_sha256': manifest['sha256'], 'source_sha256': source_files})
        print(json.dumps({'status': 'completed', 'frames': manifest['frames'],
                          'sha256': manifest['sha256']}))
    finally:
        journal.close()


if __name__ == '__main__':
    main()
