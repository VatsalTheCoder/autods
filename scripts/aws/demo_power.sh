#!/usr/bin/env bash
#
# Start and stop the demo instance. BUILD_PLAN section 11 asks for "a written
# stop/start procedure for demo days"; a script is a written procedure that
# cannot be misremembered at 9am.
#
#   ./scripts/aws/demo_power.sh start    # ~2 min to a usable URL
#   ./scripts/aws/demo_power.sh stop     # billing for compute stops
#   ./scripts/aws/demo_power.sh status
#
# Set AUTODS_INSTANCE_ID (and optionally AWS_REGION) first, or pass the id as a
# second argument.
#
# NOT YET RUN against a real account -- see docs/DEPLOYMENT.md.
#
# Why stopping matters: a stopped instance bills nothing for compute, but the
# EBS volume is still charged -- so this reduces the bill, it does not zero it.
# The database and its embeddings survive a stop, which is the point; only a
# terminate destroys them.
#
# The public IP changes on every start unless an Elastic IP is attached, so the
# URL you circulated yesterday is not the URL today. The script prints the
# current one rather than assuming.

set -euo pipefail

ACTION="${1:?usage: demo_power.sh <start|stop|status> [instance-id]}"
INSTANCE_ID="${2:-${AUTODS_INSTANCE_ID:?set AUTODS_INSTANCE_ID or pass the instance id}}"
REGION="${AWS_REGION:-eu-west-2}"

describe() {
  aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${REGION}" \
    --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' \
    --output text
}

case "${ACTION}" in
  start)
    echo "==> Starting ${INSTANCE_ID}"
    aws ec2 start-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" >/dev/null
    aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${REGION}"
    read -r STATE IP <<<"$(describe)"
    echo "    ${STATE}, public IP ${IP}"
    echo
    echo "The containers restart themselves (restart: unless-stopped), but the"
    echo "app is not up the moment the instance is. Wait for health before"
    echo "sending anyone the link:"
    echo "  curl -fsS http://${IP}:8501/_stcore/health && echo ' <- ready'"
    echo
    echo "Then: http://${IP}:8501"
    ;;
  stop)
    echo "==> Stopping ${INSTANCE_ID}"
    aws ec2 stop-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" >/dev/null
    aws ec2 wait instance-stopped --instance-ids "${INSTANCE_ID}" --region "${REGION}"
    echo "    stopped. Compute is no longer billing; the EBS volume still is."
    ;;
  status)
    read -r STATE IP <<<"$(describe)"
    echo "${INSTANCE_ID}: ${STATE} ${IP}"
    ;;
  *)
    echo "unknown action '${ACTION}' -- expected start, stop or status" >&2
    exit 2
    ;;
esac
