# Deploying AutoDS to AWS

Section 11 of [BUILD_PLAN.md](../BUILD_PLAN.md), spec milestone M7.

## Status: written, not yet executed

**No part of this has been run against a real AWS account.** There is no
instance, no URL, and no bill. That is a deliberate stopping point, not an
omission in progress — the account has not been paid for.

Being precise about what that means, because "documented" and "working" are
very different claims and this document is worthless if it blurs them:

| | |
|---|---|
| **Verified** | Everything that runs locally: the production compose file parses, refuses to start without its secrets, and the Secrets Manager source is covered by 13 tests including two mutation checks |
| **Reasoned, unverified** | Every step that needs an AWS account: the instance, the IAM role, the secret, the alarm, the bootstrap script end to end |

Treat the AWS-side commands as a considered first draft. The failure modes
called out inline are the ones worth knowing in advance; the ones nobody
predicts are exactly what the first real run is for.

What the account *would* cost, so the decision is informed: a t3.medium is about
$0.042/hour in eu-west-2, so roughly **$30/month left running** or about **$1
for a day of demoing** if stopped afterwards. The 20 GB EBS root volume is about
$1.60/month whether the instance runs or not. S3 and Secrets Manager at this
volume are cents. The Gemini free tier keeps model inference at $0.00 —
[RUNBOOK.md](RUNBOOK.md) has the measured token figures.

## What is already true

Section 0 built for this, and two of Section 11's bullets need no work at all:

- **Nothing writes to local disk.** Artifacts go through `app/core/storage.py`,
  which talks to MinIO locally and real S3 on AWS. The only difference is
  whether `S3_ENDPOINT_URL` is set.
- **All configuration comes from the environment**, so the same image runs in
  both places.
- **The vector store needs no volume of its own.** The plan offered ChromaDB on
  a persistent EBS volume *or* the pgvector fallback; the code took pgvector
  back in Section 10, so embeddings are rows in Postgres and there is no
  seventh container to keep alive. That bullet is closed by construction.
- **The access gate exists.** `AUTODS_ACCESS_TOKEN` (Section 11's PR #25) means
  a public URL is not an open door.

One gap that this section found and fixed: there was **no way to select real S3
from the environment**. `S3_ENDPOINT_URL` defaults to the local MinIO URL, an
absent variable falls back to that default, and an empty one used to arrive as
`""` and be handed to boto3 verbatim, which rejects it. So the "deploying is a
configuration change" claim was not quite true. An empty value now means
"unset", covered by `test_empty_s3_endpoint_env_var_selects_real_aws`.

## The shape of it

One t3.medium running the same containers as your laptop, minus MinIO:

```
                internet
                    │
                    ▼  :8501 only
        ┌───────────────────────┐
        │  EC2 t3.medium        │
        │                       │
        │   ui ──► api ──► worker
        │            │       │  │
        │         postgres redis│      IAM role
        │         (pgvector)    │──────────────► S3 (artifacts)
        └───────────────────────┘──────────────► Secrets Manager
```

**Only 8501 is published.** Postgres, Redis and the API listen on the compose
network only. The API is deliberately unreachable from outside: the access gate
lives in the UI, so publishing 8000 would route straight around it.

RDS, ElastiCache and ECS Fargate are future work — see the end of this document.

### Sizing

t3.medium (2 vCPU, 4 GB) is the floor, and the constraint is memory, not CPU.
The peak is a worker holding a training frame while SHAP holds an explanation
over it; [RUNBOOK.md](RUNBOOK.md) records a 100k-row run at about four minutes,
most of it EDA. `--concurrency=2` in the production compose file is already
optimistic for 4 GB — two concurrent jobs can both be at their peak. Raise it
only after watching `docker stats` during a real pair of runs, and prefer a
t3.large to a higher concurrency on the same box.

t3 instances are burstable. A long pipeline can exhaust CPU credits and quietly
get slower rather than failing, which looks like the model being slow. If runs
degrade over a demo session, check `CPUCreditBalance` before blaming the agents.

## Steps

Set these once:

```bash
export AWS_REGION=eu-west-2
export S3_BUCKET=autods-artifacts-$RANDOM   # bucket names are globally unique
```

### 1. The artifact bucket

```bash
aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION" \
  --create-bucket-configuration LocationConstraint="$AWS_REGION"

aws s3api put-public-access-block --bucket "$S3_BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

Uploaded datasets and generated reports go here. They are served through the API,
never by a public bucket URL, so the block is right.

`--create-bucket-configuration` is required in every region except us-east-1,
where passing it is an error. If you deploy to us-east-1, drop that line.

### 2. The secret

```bash
aws secretsmanager create-secret \
  --name autods/production \
  --description "AutoDS runtime secrets" \
  --secret-string '{"google_api_key":"AQ...your-key..."}' \
  --region "$AWS_REGION"
```

The name (`autods/production`) is what `AWS_SECRETS_ID` refers to. The name is
not sensitive; the contents are.

Any key in the JSON that matches a setting in `app/core/config.py` is applied —
lower case or `SCREAMING_CASE`, both work. Keys that are not settings are
ignored, so a secret shared with something else is harmless.

**An environment variable beats the secret.** That is deliberate, so an override
on the box does not need a redeploy, but it means a stale variable can mask a
rotated key. After rotating, confirm nothing is shadowing it:

```bash
docker compose -f docker-compose.prod.yml exec api env | grep -i google
```

Nothing should come back.

### 3. The IAM role

The instance gets a role, so no long-lived access key exists to leak. boto3
finds it automatically, which is why the production compose file sets no
`AWS_ACCESS_KEY_ID`.

Trust policy:

```bash
cat > /tmp/trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON

aws iam create-role --role-name autods-instance \
  --assume-role-policy-document file:///tmp/trust.json
```

Permissions — the bucket and that one secret, nothing else:

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
cat > /tmp/policy.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject"],
  "Resource":"arn:aws:s3:::${S3_BUCKET}/*"},
 {"Effect":"Allow","Action":["s3:ListBucket"],
  "Resource":"arn:aws:s3:::${S3_BUCKET}"},
 {"Effect":"Allow","Action":["secretsmanager:GetSecretValue"],
  "Resource":"arn:aws:secretsmanager:${AWS_REGION}:${ACCOUNT}:secret:autods/production-*"}]}
JSON

aws iam put-role-policy --role-name autods-instance \
  --policy-name autods-runtime --policy-document file:///tmp/policy.json

aws iam create-instance-profile --instance-profile-name autods-instance
aws iam add-role-to-instance-profile --instance-profile-name autods-instance \
  --role-name autods-instance
```

The `-*` suffix on the secret ARN is not a typo. Secrets Manager appends six
random characters to every ARN, so an exact-match resource never matches.

`ListBucket` is on the bucket ARN and the object actions are on `/*`. Getting
that pair wrong produces `AccessDenied` on listing while individual reads work,
which is a confusing way to find out.

### 4. The instance

t3.medium, Ubuntu 24.04, 20 GB gp3, the instance profile above, and a security
group opening 8501 and 22. Restrict both to your own IP if you can — the access
token is a shared secret, not authentication, and SSH open to the world will be
found within the hour.

Then, over SSH:

```bash
sudo REPO_URL=https://github.com/VatsalTheCoder/autods.git \
  bash /opt/autods/scripts/aws/bootstrap.sh
```

[`bootstrap.sh`](../scripts/aws/bootstrap.sh) installs Docker from Docker's own
repository (Ubuntu's `docker.io` has no `docker compose` plugin), clones the
repo, writes `/etc/autods/autods.env` as a root-owned mode-600 template, builds
the image, runs migrations *before* starting the app, and waits for health. It
is idempotent, so the first run stopping at "fill in the template" is expected —
fill it in and run it again.

It deliberately does not take secrets as arguments. Passing them through
user-data would put them in the instance metadata service, readable by anything
on the box that can reach `169.254.169.254`.

#### The one bootstrap secret

`POSTGRES_PASSWORD` lives in `/etc/autods/autods.env` rather than Secrets
Manager, which is a real inconsistency and worth naming. Postgres has to be
running with its password set before anything can connect to it, and the
password would otherwise have to be read from AWS by the very container that
needs it to start. Everything else — the API key above all — comes from Secrets
Manager. Generate it with `openssl rand -base64 24` and never reuse the local
`autods`/`autods` pair.

### 5. The billing alarm

```bash
./scripts/aws/billing_alarm.sh you@example.com 5
```

Two ways this silently never fires, both handled in
[`billing_alarm.sh`](../scripts/aws/billing_alarm.sh) and both worth repeating:
billing metrics exist **only in us-east-1** regardless of where the instance is,
and **"Receive Billing Alerts" must be enabled in the Billing console** — it
cannot be turned on from the CLI, and until it is, the alarm sits in
`INSUFFICIENT_DATA` looking healthy.

Confirm the SNS subscription email. An unconfirmed subscription is dropped.

### 6. Demo days

```bash
export AUTODS_INSTANCE_ID=i-0123456789abcdef0
./scripts/aws/demo_power.sh start     # ~2 min
./scripts/aws/demo_power.sh stop      # afterwards
```

A stopped instance bills no compute; the EBS volume is still charged. The
database and its embeddings survive a stop — only terminating destroys them.

**The public IP changes on every start** unless you attach an Elastic IP, so
yesterday's URL is not today's. `demo_power.sh start` prints the current one.
Attach an Elastic IP if the URL is going to anyone else; note that an unattached
Elastic IP is itself billed, which is a classic surprise on a stopped instance.

Start it a few minutes early. The containers come back on their own, but the
first pipeline run pays model latency — budget the minute of dead air that
[RUNBOOK.md](RUNBOOK.md) measured.

## Verifying it actually works

An app that loads is not an app that runs. In order:

```bash
curl -fsS http://<ip>:8501/_stcore/health          # UI up
docker compose -f docker-compose.prod.yml exec api curl -fsS localhost:8000/health
```

The API's `/health` reports storage and database separately, so it distinguishes
"running" from "running and able to reach S3" — the single most likely thing to
be wrong on a first deploy, and it will be an IAM policy rather than the code.

Then upload `data/examples/customer_churn.csv` through the UI and let it finish.
That exercises S3 (upload and artifacts), Postgres, Redis, Celery, the model
API, and PDF rendering. Nothing short of a completed run proves the deployment;
each of those has failed independently at some point in this project's history.

## Future work

**RDS + ElastiCache instead of containers.** Managed Postgres and Redis get
backups, patching and failover. The application needs no change — both are
already `DATABASE_URL` and `REDIS_URL`. The reason it is not the starting point
is cost: RDS roughly doubles the monthly bill for a project that is demonstrated
occasionally, and a `docker compose` Postgres on EBS loses nothing that matters
at this scale. Worth doing the moment anyone else's data is in it.

**ECS Fargate instead of EC2.** The containers are stateless apart from
Postgres, so api/worker/ui move to Fargate services more or less as they are:
no instance to patch, and the worker scales independently of the API — the right
shape, since the worker is where the time goes. It needs the compose file
translated into task definitions, a load balancer in front of the UI, and
Postgres moved to RDS first. That is a Section 11 rewrite, not a configuration
change, which is why the plan asks for it to be *documented* rather than built.

**HTTPS.** The UI is HTTP on 8501, so the access token crosses the network in
clear text. Caddy or an ALB in front with a real certificate is the fix. Fine
for a demo to a known audience over a restricted security group; not fine for
anything longer-lived.

**Zero-downtime deploys.** `docker compose up -d` after a `git pull` drops
requests. Nobody is served during a demo-only deployment, so this is noted
rather than solved.
