#!/usr/bin/env bash
#
# Create a billing alarm that emails when the estimated AWS charge crosses a
# threshold. BUILD_PLAN section 11 asks for "an alarm that emails you if the
# bill goes above a few pounds", and this is that, in the smallest form that
# actually works.
#
#   ./scripts/aws/billing_alarm.sh you@example.com 5
#
# NOT YET RUN against a real account -- see docs/DEPLOYMENT.md.
#
# Two things about billing metrics that are easy to get wrong and give you an
# alarm that silently never fires:
#
#   1. AWS/Billing metrics exist ONLY in us-east-1, whatever region the
#      infrastructure is in. Every call below pins that region deliberately;
#      do not "fix" it to match the instance's region.
#   2. "Receive Billing Alerts" must be switched on in the Billing console
#      (Billing preferences -> Alert preferences). It cannot be enabled from the
#      CLI, and until it is, EstimatedCharges is never published -- so the alarm
#      sits in INSUFFICIENT_DATA for ever and looks like it is working.
#
# EstimatedCharges is also cumulative across the calendar month and resets on
# the 1st, so this alarms on the monthly total, not on a daily rate.

set -euo pipefail

EMAIL="${1:?usage: billing_alarm.sh <email> [threshold-usd]}"
THRESHOLD="${2:-5}"
REGION="us-east-1"
TOPIC_NAME="autods-billing"
ALARM_NAME="autods-estimated-charges"

echo "==> Creating (or reusing) SNS topic ${TOPIC_NAME} in ${REGION}"
# create-topic is idempotent: it returns the existing ARN if the topic is there.
TOPIC_ARN="$(aws sns create-topic \
  --name "${TOPIC_NAME}" \
  --region "${REGION}" \
  --query 'TopicArn' --output text)"
echo "    ${TOPIC_ARN}"

echo "==> Subscribing ${EMAIL}"
aws sns subscribe \
  --topic-arn "${TOPIC_ARN}" \
  --protocol email \
  --notification-endpoint "${EMAIL}" \
  --region "${REGION}" >/dev/null
echo "    Confirm the link in that inbox. An unconfirmed subscription is"
echo "    silently dropped, which is the usual reason an alarm 'does not email'."

echo "==> Creating alarm ${ALARM_NAME} at \$${THRESHOLD}"
aws cloudwatch put-metric-alarm \
  --alarm-name "${ALARM_NAME}" \
  --alarm-description "AutoDS estimated monthly charges above \$${THRESHOLD}" \
  --namespace "AWS/Billing" \
  --metric-name "EstimatedCharges" \
  --dimensions Name=Currency,Value=USD \
  --statistic Maximum \
  --period 21600 \
  --evaluation-periods 1 \
  --threshold "${THRESHOLD}" \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "${TOPIC_ARN}" \
  --region "${REGION}"

echo
echo "Done. Verify with:"
echo "  aws cloudwatch describe-alarms --alarm-names ${ALARM_NAME} --region ${REGION} \\"
echo "    --query 'MetricAlarms[0].StateValue' --output text"
echo
echo "INSUFFICIENT_DATA immediately after creation is expected -- the metric"
echo "publishes roughly every six hours. Still INSUFFICIENT_DATA a day later"
echo "means billing alerts are not enabled; see the note at the top of this file."
