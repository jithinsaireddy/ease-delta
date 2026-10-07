# Security

## Reporting

Please report vulnerabilities privately through GitHub's **Report a vulnerability** button on the repository's
Security tab, not in a public issue. You should hear back within a week.

## What to know when deploying

- `ease serve` without `--multi` has **no authentication** and binds to 127.0.0.1. Do not expose it to a network.
- With `--multi`, a workspace key is a bearer token: anyone holding it is that workspace. Keys are stored only as
  hashes; rotate one that may have leaked (`ease workspace rotate-key`).
- Put a TLS-terminating reverse proxy in front of any server reachable by others; the service speaks plain HTTP.
- Confirmation links are single-use and expire, but anyone with a link can answer it.
- Imported files are parsed, never executed. Model checkpoints are loaded as safetensors, and training checkpoints
  with `weights_only=True`.
- Nothing in the code can send messages, make payments or execute actions.
