"""Single entry point for the two kinds of result processing (stress tests).

  python -m report aiperf  <results-dir> --target 30 [--gpu 8] [--require-metrics] [--ttft-max ms]
  python -m report agents  <results-dir> --target 30 [--gpu 8] [--require-metrics]

--gpu is optional: without it, the number of GPUs is deduced, run by run, from the fingerprint stored in the
server_config.json of the run (TP x PP x DP x number of vLLM servers). Give it to force the value.
Observation verdicts (Prometheus) are not produced here: use `python -m observation.query`."""
import argparse

from report import agents, aiperf

KINDS = {"aiperf": aiperf, "agents": agents}


def build_parser():
    """Build the argument parser, with one sub-command per kind of results (aiperf, agents)."""
    parser = argparse.ArgumentParser(prog="python -m report", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="type", required=True, metavar="{" + ",".join(KINDS) + "}")
    for name, module in KINDS.items():
        sp = sub.add_parser(name, help=module.HELP, description=module.__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
        sp.add_argument("root", help="results directory")
        sp.add_argument("--target", type=float, required=True, help="target interactivity per user (tokens/s)")
        sp.add_argument("--gpu", type=int, default=None,
                        help="number of GPUs behind the target (all replicas); default: deduced from the fingerprint "
                             "in each run's server_config.json (TP x PP x DP x vLLM servers)")
        module.add_arguments(sp)
        sp.set_defaults(execute=module.execute)
    return parser


def main(argv=None):
    """Parse `argv` (default: sys.argv) and run the chosen kind."""
    args = build_parser().parse_args(argv)
    args.execute(args)
