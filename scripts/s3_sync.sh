#!/usr/bin/env bash
# Move data between a local mirror and the project bucket (handoff §4.3).
#
#   scripts/s3_sync.sh pull                  # bucket -> ~/psplat-data, without renders
#   scripts/s3_sync.sh pull --with-renders   # including eval renders
#   scripts/s3_sync.sh push-trained DIR SCENE TRAINER SHA
#   scripts/s3_sync.sh push-run DIR TAG     # to runs/<date>-<tag>/
#
# Bucket layout:
#   raw/<set>/<scene>/                      datasets, immutable
#   trained/<scene>/<trainer>/<sha>/        ply + cams + renders
#   runs/<date>-<tag>/                      weights, ordered, packed, results.parquet
#   study/                                  stimuli and anonymised responses
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/aws_lib.sh
source "$here/aws_lib.sh"
require aws
bucket="$(bucket_name)"
mirror="${PSPLAT_DATA:-$HOME/psplat-data}"

case "${1:-}" in
  pull)
    exclude=(--exclude "*/renders/*")
    [ "${2:-}" = "--with-renders" ] && exclude=()
    mkdir -p "$mirror"
    aws s3 sync "s3://$bucket/" "$mirror/" ${exclude[@]+"${exclude[@]}"}
    ;;
  push-trained)
    [ $# -eq 5 ] || die "usage: $0 push-trained DIR SCENE TRAINER SHA"
    aws s3 sync "$2" "s3://$bucket/trained/$3/$4/$5/"
    ;;
  push-run)
    [ $# -eq 3 ] || die "usage: $0 push-run DIR TAG"
    [ -f "$2/run.json" ] || die "$2 has no run.json; no result without provenance"
    aws s3 sync "$2" "s3://$bucket/runs/$(date -u +%Y-%m-%d)-$3/"
    ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
