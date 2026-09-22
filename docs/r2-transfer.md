# R2 image relay

RemoteClaudeBackend defaults to Cloudflare R2. Configuration is loaded from
`~/.config/clothagent/r2.env` without executing shell commands. Existing environment
variables take precedence (never commit credentials):

- R2_ACCOUNT_ID
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

Install project dependencies, including boto3. The R2 token needs object read/write
access to the chosen private bucket. No R2 credentials are sent to the planner.
Boto3 signs PUT and GET URLs locally; curl performs the PUT with the existing
30-second attempt timeout and bounded retries. GET URLs expire after one hour.
The remote host downloads with curl and verifies SHA256. Missing configuration
fails explicitly; there is no automatic fallback to the old relay.

A user systemd timer `clothagent-r2-cleanup.timer` runs `scripts/cleanup_r2.py`
every five minutes. It deletes only `cloth-agent/<32 hex digits>.png` objects whose
LastModified is at least one hour old. Normal retention is 60–65 minutes; network
failures or the user service manager being offline delay deletion until recovery.
It runs independently of the fold loop. The token needs list/delete permissions
in addition to read/write. URL expiry alone does not delete stored objects.
Restart the fold process to load the new configuration handling.

Each uncached PNG costs one PUT and one GET when downloaded successfully.
Signing URLs and local/remote cache hits do not make R2 API requests. Retries add
requests. Recent full iteration evidence counts: 3 + 1 + 6 + 6 + 9 + 4 = 29
image references before deduplication/cache reuse, or about 29 PUT + 29 GET
without caching or retries. These are request counts, not billing guarantees.
