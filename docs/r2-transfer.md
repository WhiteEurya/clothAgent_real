# R2 image relay

RemoteClaudeBackend defaults to Cloudflare R2. Set these variables in the environment
that launches the fold loop (never commit credentials):

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

Configure a bucket lifecycle rule to delete objects under `cloth-agent/` after
one day. Signed URL expiration does not delete stored objects. No lifecycle rule
is created automatically. Restart the fold process after configuring the environment.

Each uncached PNG costs one PUT and one GET when downloaded successfully.
Signing URLs and local/remote cache hits do not make R2 API requests. Retries add
requests. Recent full iteration evidence counts: 3 + 1 + 6 + 6 + 9 + 4 = 29
image references before deduplication/cache reuse, or about 29 PUT + 29 GET
without caching or retries. These are request counts, not billing guarantees.
