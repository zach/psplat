#!/usr/bin/env bash
# Provision psplat's AWS GPU box from the Mac (handoff §5.1). Idempotent: each
# resource is created only if missing, and an existing instance is started
# rather than duplicated.
#
#   scripts/aws_provision.sh            # bucket, role, security group, instance
#   scripts/aws_provision.sh --no-launch  # everything except the instance
#
# Creates, all tagged project=psplat:
#   - S3 bucket (private, versioned, runs/ moves to Infrequent Access at 30 days)
#   - IAM role and instance profile: SSM access plus read-write to that bucket only
#   - security group with no inbound rules; access is SSM Session Manager only
#   - the instance: Deep Learning Base GPU AMI (Amazon Linux 2023), gp3 root volume
#   - optionally a monthly budget alarm (PSPLAT_ALERT_EMAIL)
#
# Launching bills by the hour, so it asks first unless PSPLAT_YES=1. Settings
# are environment variables; see scripts/aws_lib.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/aws_lib.sh
source "$here/aws_lib.sh"

launch=1
[ "${1:-}" = "--no-launch" ] && launch=0

require aws
account="$(account_id)"
bucket="$(bucket_name)"
log "account $account, region $AWS_REGION, bucket $bucket"

# Bucket --------------------------------------------------------------------------
if aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
  log "bucket exists"
else
  log "creating bucket"
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$bucket" >/dev/null
  else
    aws s3api create-bucket --bucket "$bucket" \
      --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
  fi
fi
aws s3api put-public-access-block --bucket "$bucket" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-versioning --bucket "$bucket" --versioning-configuration Status=Enabled
aws s3api put-bucket-tagging --bucket "$bucket" --tagging 'TagSet=[{Key=project,Value=psplat}]'
aws s3api put-bucket-lifecycle-configuration --bucket "$bucket" --lifecycle-configuration '{
  "Rules": [{"ID": "runs-to-ia", "Status": "Enabled", "Filter": {"Prefix": "runs/"},
             "Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}]}]}'

# IAM role --------------------------------------------------------------------------
if aws iam get-role --role-name "$PSPLAT_ROLE" >/dev/null 2>&1; then
  log "role exists"
else
  log "creating role $PSPLAT_ROLE"
  aws iam create-role --role-name "$PSPLAT_ROLE" --tags "$PSPLAT_TAGS" \
    --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}' >/dev/null
fi
aws iam attach-role-policy --role-name "$PSPLAT_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name "$PSPLAT_ROLE" --policy-name psplat-bucket --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"s3:ListBucket\"], \"Resource\": \"arn:aws:s3:::$bucket\"},
    {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\", \"s3:DeleteObject\"],
     \"Resource\": \"arn:aws:s3:::$bucket/*\"}]}"
if ! aws iam get-instance-profile --instance-profile-name "$PSPLAT_ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PSPLAT_ROLE" --tags "$PSPLAT_TAGS" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PSPLAT_ROLE" --role-name "$PSPLAT_ROLE"
  log "waiting for the instance profile to propagate"
  sleep 15
fi

# Security group: no inbound rules at all ---------------------------------------------
vpc="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
[ "$vpc" != "None" ] || die "no default VPC in $AWS_REGION; create one or extend this script"
sg="$(aws ec2 describe-security-groups --filters Name=group-name,Values=psplat-ssm-only \
  Name=vpc-id,Values="$vpc" --query 'SecurityGroups[0].GroupId' --output text)"
if [ "$sg" = "None" ]; then
  sg="$(aws ec2 create-security-group --group-name psplat-ssm-only --vpc-id "$vpc" \
    --description "psplat: no inbound; SSM Session Manager only" \
    --tag-specifications "ResourceType=security-group,Tags=[{$PSPLAT_TAGS}]" \
    --query GroupId --output text)"
  log "created security group $sg"
fi

# Budget alarm (optional) ---------------------------------------------------------------
if [ -n "${PSPLAT_ALERT_EMAIL:-}" ]; then
  if aws budgets describe-budget --account-id "$account" --budget-name psplat >/dev/null 2>&1; then
    log "budget exists"
  else
    log "creating monthly budget alarm: $PSPLAT_BUDGET_USD USD, alerts at 80% to $PSPLAT_ALERT_EMAIL"
    aws budgets create-budget --account-id "$account" \
      --budget "{\"BudgetName\": \"psplat\", \"BudgetType\": \"COST\", \"TimeUnit\": \"MONTHLY\",
        \"BudgetLimit\": {\"Amount\": \"$PSPLAT_BUDGET_USD\", \"Unit\": \"USD\"},
        \"CostFilters\": {\"TagKeyValue\": [\"user:project\$psplat\"]}}" \
      --notifications-with-subscribers "[{\"Notification\": {\"NotificationType\": \"ACTUAL\",
        \"ComparisonOperator\": \"GREATER_THAN\", \"Threshold\": 80, \"ThresholdType\": \"PERCENTAGE\"},
        \"Subscribers\": [{\"SubscriptionType\": \"EMAIL\", \"Address\": \"$PSPLAT_ALERT_EMAIL\"}]}]"
    log "activate the project cost-allocation tag in the Billing console for the filter to apply"
  fi
else
  log "no PSPLAT_ALERT_EMAIL: skipping the budget alarm"
fi

[ "$launch" = "1" ] || { log "--no-launch: done"; exit 0; }

# Instance ------------------------------------------------------------------------------
existing="$(find_instance)"
if [ -n "$existing" ]; then
  state="$(aws ec2 describe-instances --instance-ids "$existing" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"
  log "instance $existing already exists ($state)"
  if [ "$state" = "stopped" ]; then
    confirm "Start $existing ($PSPLAT_INSTANCE_TYPE)? It bills by the hour while running." \
      || die "not starting"
    aws ec2 start-instances --instance-ids "$existing" >/dev/null
    log "started; connect with: aws ssm start-session --target $existing"
  fi
  exit 0
fi

ami="$(resolve_ami)"
# macOS ships bash 3.2, where "${a[@]}" of an empty array fails under set -u;
# the ${a[@]+...} form below is the portable spelling.
market=()
if [ "${PSPLAT_SPOT:-}" = "1" ]; then
  market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}')
fi
confirm "Launch $PSPLAT_INSTANCE_TYPE${PSPLAT_SPOT:+ (Spot)} from $ami with ${PSPLAT_VOLUME_GB} GB gp3? It bills by the hour while running." \
  || die "not launching"
instance="$(aws ec2 run-instances --image-id "$ami" --instance-type "$PSPLAT_INSTANCE_TYPE" \
  --iam-instance-profile "Name=$PSPLAT_ROLE" --security-group-ids "$sg" \
  --metadata-options HttpTokens=required \
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$PSPLAT_VOLUME_GB,VolumeType=gp3,Iops=6000,Throughput=500,DeleteOnTermination=true}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$PSPLAT_NAME},{$PSPLAT_TAGS}]" \
    "ResourceType=volume,Tags=[{$PSPLAT_TAGS}]" \
  --instance-initiated-shutdown-behavior stop \
  ${market[@]+"${market[@]}"} --query 'Instances[0].InstanceId' --output text)"
log "launched $instance; waiting until it is running"
aws ec2 wait instance-running --instance-ids "$instance"
cat >&2 <<EOF
[psplat] ready. Next:
[psplat]   aws ssm start-session --target $instance
[psplat]   then on the instance:
[psplat]   git clone https://github.com/zach/psplat ~/psplat && ~/psplat/scripts/aws_bootstrap.sh
EOF
