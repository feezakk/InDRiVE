#!/usr/bin/env bash
# CI helper: verify pretrain helper sets intrinsic-only flags (disag=1.0, extr=0.0)
set -euo pipefail
script=tools/pretrain_latent_disagreement.sh
if ! grep -q "--dreamerv3.expl_rewards.disag 1.0" "$script"; then
  echo "ERROR: $script does not enable disag intrinsic reward flag (--dreamerv3.expl_rewards.disag 1.0)"
  exit 2
fi
if ! grep -q "--dreamerv3.expl_rewards.extr 0.0" "$script"; then
  echo "ERROR: $script does not disable extrinsic reward (--dreamerv3.expl_rewards.extr 0.0)"
  exit 3
fi
# Everything looks good
echo "$script: PASS"
exit 0
