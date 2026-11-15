"""Single entry point for the three kinds of result processing.

  python -m report aiperf       <results-dir> --target 30 --gpu 8 [--require-metrics] [--ttft-max ms]
  python -m report agents       <results-dir> --target 30 --gpu 8 [--require-metrics]
  python -m report observation  <results-dir> --target 30 --gpu 8 [--ttft-max ms]
                                              [--view windows|period|agents] [--period hour|range]
"""
import argparse

from report import agents, aiperf, observation

KINDS = {"aiperf": aiperf, "agents": agents, "observation": observation}


def build_parser():
    """Build the argument parser, with one sub-command per kind of results (aiperf, agents, observation)."""
    parser = argparse.ArgumentParser(prog="python -m report", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="type", required=True, metavar="{" + ",".join(KINDS) + "}")
    for name, module in KINDS.items():
        sp = sub.add_parser(name, help=module.HELP, description=module.__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
        sp.add_argument("root", help="results directory")
        sp.add_argument("--target", type=float, required=True, help="target interactivity per user (tokens/s)")
        sp.add_argument("--gpu", type=int, required=True, help="number of GPUs behind the target (all replicas)")
        module.add_arguments(sp)
        sp.set_defaults(execute=module.execute)
    return parser


def main(argv=None):
    """Parse `argv` (default: sys.argv) and run the chosen kind."""
    args = build_parser().parse_args(argv)
    args.execute(args)
