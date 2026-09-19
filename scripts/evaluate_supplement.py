"""Run the independent, versioned short stress and shared-population protocol."""
import argparse
import json
from pathlib import Path
import signal

from neuroterrarium.supplemental import evaluate_supplement, summarize_supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/evaluate-supplement-v1.json'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--models', type=Path, default=Path('artifacts/models'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize_supplement(args.output)
        return 0
    if args.data is None:
        parser.error('--data is required to run the complete graph')
    cancelled = False

    def cancel(_signal, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    result = evaluate_supplement(json.loads(args.config.read_text()), args.data, args.models,
                                 args.output, cancel=lambda: cancelled)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] in {'completed', 'verified'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
