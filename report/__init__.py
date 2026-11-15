"""Processing of the bench results: one sub-command per result type (aiperf, agents).

Run from the repository root:
  python -m report aiperf  <results-dir> --target 30 [--gpu 8]
  python -m report agents  <results-dir> --target 30 [--gpu 8]
--gpu is optional (deduced from the fingerprint in server_config.json). Observation verdicts come from
`python -m observation.query`.
Standard library only."""
