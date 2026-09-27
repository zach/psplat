# shellcheck shell=bash
# Shared settings and helpers for the psplat AWS scripts. Source, do not run.
#
# Everything is configured through environment variables, so no account
# detail lives in the repository:
#
#   PSPLAT_AWS_REGION      default us-east-1 (decision D2)
#   PSPLAT_BUCKET          default psplat-<account id>
#   PSPLAT_INSTANCE_TYPE   default g6e.2xlarge (1x L40S 48 GB); g6.xlarge for smoke runs
#   PSPLAT_NAME            default psplat-gpu; the instance's Name tag
#   PSPLAT_VOLUME_GB       default 500
#   PSPLAT_AMI_SSM_PARAM   pin the AMI's public SSM parameter instead of discovering it
#   PSPLAT_SPOT            set to 1 for a Spot instance (batch benches with checkpoints)
#   PSPLAT_ALERT_EMAIL     if set, a monthly budget alarm is created for this address
#   PSPLAT_BUDGET_USD      default 200; the budget alarm's monthly limit

export AWS_REGION="${PSPLAT_AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"
PSPLAT_NAME="${PSPLAT_NAME:-psplat-gpu}"
PSPLAT_INSTANCE_TYPE="${PSPLAT_INSTANCE_TYPE:-g6e.2xlarge}"
PSPLAT_VOLUME_GB="${PSPLAT_VOLUME_GB:-500}"
PSPLAT_BUDGET_USD="${PSPLAT_BUDGET_USD:-200}"
# Used by the scripts that source this file.
# shellcheck disable=SC2034
PSPLAT_ROLE="psplat-instance"
# shellcheck disable=SC2034
PSPLAT_TAGS="Key=project,Value=psplat"
DLAMI_PREFIX="/aws/service/deeplearning/ami/x86_64"

log() { printf '[psplat] %s\n' "$*" >&2; }
die() { printf '[psplat] error: %s\n' "$*" >&2; exit 1; }

require() {
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is required but not on PATH"
  done
}

account_id() { aws sts get-caller-identity --query Account --output text; }

bucket_name() {
  if [ -n "${PSPLAT_BUCKET:-}" ]; then
    printf '%s\n' "$PSPLAT_BUCKET"
  else
    printf 'psplat-%s\n' "$(account_id)"
  fi
}

# Read public SSM parameter names on stdin, one per line, and print the one
# for the Deep Learning Base GPU AMI on Amazon Linux 2023 (x86_64, latest).
# Fails, listing the candidates, unless exactly one matches, so a renamed or
# newly added flavour is noticed instead of silently picked.
pick_dlami_param() {
  local matches count
  matches="$(grep -E '/latest/ami-id$' \
    | grep -i 'base' \
    | grep -i 'amazon-linux-2023' \
    | grep -iv -e 'arm64' -e 'neuron' -e 'pytorch' -e 'tensorflow' || true)"
  count="$(printf '%s' "$matches" | grep -c . || true)"
  if [ "$count" = "1" ]; then
    printf '%s\n' "$matches"
    return 0
  fi
  if [ "$count" = "0" ]; then
    log "no Deep Learning Base AMI (Amazon Linux 2023) parameter found"
  else
    log "several candidate AMI parameters; set PSPLAT_AMI_SSM_PARAM to one of:"
    printf '%s\n' "$matches" | sed 's/^/[psplat]   /' >&2
  fi
  return 1
}

# The AMI ID to launch, resolved at run time (handoff §5.1: do not hard-code).
resolve_ami() {
  local param
  if [ -n "${PSPLAT_AMI_SSM_PARAM:-}" ]; then
    param="$PSPLAT_AMI_SSM_PARAM"
  else
    param="$(aws ssm get-parameters-by-path --path "$DLAMI_PREFIX" --recursive \
      --query 'Parameters[].Name' --output text | tr '\t' '\n' | pick_dlami_param)" \
      || die "could not choose an AMI parameter"
  fi
  log "AMI parameter: $param"
  aws ssm get-parameter --name "$param" --query Parameter.Value --output text
}

# Instance ID of the project's instance, if one exists and is not terminated.
find_instance() {
  aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$PSPLAT_NAME" "Name=tag:project,Values=psplat" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

confirm() {
  if [ "${PSPLAT_YES:-}" = "1" ]; then
    return 0
  fi
  printf '%s [y/N] ' "$1" >&2
  local answer
  read -r answer
  [ "$answer" = "y" ] || [ "$answer" = "Y" ]
}
