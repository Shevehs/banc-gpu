"""Processing of the bench results: one sub-command per result type (aiperf, agents, observation).

Run from the repository root:
  python -m report aiperf       <results-dir> --target 30 --gpu 8
  python -m report agents       <results-dir> --target 30 --gpu 8
  python -m report observation  <results-dir> --target 30 --gpu 8
Standard library only."""
