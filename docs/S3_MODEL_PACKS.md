# s3.cloud.ru storage — Mockingbird model packs

The Mockingbird whisper model packs are mirrored to an s3.cloud.ru public
bucket alongside the GitHub Releases. The runtime prefers S3 because it is
faster from RU / CIS networks and is not subject to GitHub's 2 GiB asset
limit.

## Bucket

| Field | Value |
|-------|-------|
| Service | [Cloud.ru S3](https://cloud.ru) (S3-compatible) |
| Endpoint | `https://s3.cloud.ru` |
| Region | `ru-central-1` |
| Bucket | `mockingbird` |
| Owner tenant ID | `5b7b9b26-db4d-4489-8ce4-40cc6b531fe0` |
| Auth style | AWS SigV4 with tenant-prefixed access key ID |

## Layout

```
s3://mockingbird/
└── models/
    ├── tiny.zip            66.6 MiB
    ├── base.zip           126.9 MiB
    ├── small.zip          425.1 MiB
    ├── medium.zip           1.3 GiB
    └── large-v3-turbo.zip   1.4 GiB
```

Each archive is the same layout `scripts/build_model_pack.sh` produces:
```
Mockingbird-whisper-<size>-model.zip
└── cache/
    └── models--<hf-repo-slug>/
        ├── refs/main        (commit hash)
        ├── snapshots/<hash>/
        │   ├── model.bin    → ../../blobs/<sha256>
        │   ├── config.json  → ../../blobs/<sha256>
        │   └── ...
        └── trees/<hash>.json
```

`scripts/release.sh` and `scripts/build_model_pack.sh` upload to this
bucket automatically when `S3_ENDPOINT_URL` is set in the environment.

## Local upload (manual)

```bash
# Configure credentials (tenant ID is part of the access key ID, per Cloud.ru)
cat > ~/.aws/credentials <<EOF
[default]
aws_access_key_id = 5b7b9b26-db4d-4489-8ce4-40cc6b531fe0:REMOVED-SECRET
aws_secret_access_key = REMOVED-SECRET
EOF

cat > ~/.aws/config <<EOF
[default]
region = ru-central-1
output = json
EOF

# Upload a single pack (size = tiny, base, small, medium, large-v3-turbo)
aws s3 cp installer/Mockingbird-whisper-large-v3-turbo-model.zip \
        s3://mockingbird/models/large-v3-turbo.zip \
        --endpoint-url https://s3.cloud.ru \
        --acl public-read
```

## Why the runtime still uses GitHub today

~~Cloud.ru S3 requires AWS SigV4 on every download — anonymous GET returns
`AuthorizationQueryParametersError: missing tenant id`. Without signed
requests the app cannot fetch models from `s3.cloud.ru` directly.~~

**Resolved (2026-09-29)**: Cloud.ru exposes anonymous downloads through
the bucket's *global name* (`https://mockingbird.s3.cloud.ru/...`).
The global name is configured in the bucket settings via the Cloud.ru
UI (Edit → Global bucket name / Domain name). With PublicAccessBlock
disabled and a `PublicReadGetObject` policy in place, plain HTTPS
GETs succeed without any signing.

The runtime downloads now go to `https://mockingbird.s3.cloud.ru/models/<size>.zip`
as the **primary** source. GitHub Releases are still listed as
fallback-1 (CDN coverage outside RU/CIS), huggingface_hub stays
fallback-2 (last resort). The earlier options A/B/C are obsolete —
kept here for historical context only.

### Historical context (presigned / SigV4 / proxy — no longer needed)

Earlier in the session we tried three approaches that turned out to be
unnecessary:

#### A. Presigned URL bundle baked into the release — skipped

Presigned URLs are limited to a 7-day TTL on SigV4 (no permanent option)
and require periodic regeneration. With the global-name path working,
this became moot.

#### B. AWS SigV4 signing inside the app — skipped

Embedding access key ID + secret in the binary extracts trivially from
the `.exe`. With anonymous downloads working, the security risk is gone.

#### C. Reverse proxy in front of S3 — skipped

An nginx / Caddy / Cloudflare worker would have added a hosting
dependency for what is now a non-problem.

## Current status (resolved)

- `PublicAccessBlock`: all four flags set to `false`.
- `Bucket Policy`: `PublicReadGetObject` (Allow `s3:GetObject` +
  `s3:HeadObject` from `Principal: *`, resource `arn:aws:s3:::mockingbird/*`).
- Global bucket name set in Cloud.ru UI: `mockingbird`.
- Anonymous HEAD: `HTTP/1.1 200 OK`, `Content-Length: 69808214`,
  `Content-Type: application/zip` — verified for all five packs.

The runtime code is live (`_download_from_s3()` is the primary path,
`_S3_BASE = "https://mockingbird.s3.cloud.ru/models"`).
