#!/usr/bin/env python3
"""Write a stand-in policy handoff: a zero-action actor.onnx and a level.yaml
of adam-rl's constants with FROM adam-rl markers.

    ros2 run genesis_policy make_stub_policy.py --out /handoff/policy/stub
    ros2 launch genesis_policy policy.launch.py policy_dir:=/handoff/policy/stub dry_run:=true
"""
import argparse
import sys

from genesis_policy.stub import write_stub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='directory to write actor.onnx and level.yaml into')
    ap.add_argument('--no-actor', action='store_true', help='write level.yaml only')
    args = ap.parse_args()
    level = write_stub(args.out, with_actor=not args.no_actor)
    print(f'wrote {args.out}/level.yaml ({level.layout.size} observations, {level.rate_hz:g} Hz)'
          + ('' if args.no_actor else f' and {args.out}/actor.onnx (zero action)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
