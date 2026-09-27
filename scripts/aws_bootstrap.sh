#!/usr/bin/env bash
# Bootstrap the AWS GPU box (Deep Learning Base GPU AMI, Amazon Linux 2023).
# Run on the instance, from a psplat checkout; safe to run again.
#
#   ~/psplat/scripts/aws_bootstrap.sh
#
# Installs system tools, uv and Python 3.12, a venv at ~/.venvs/psplat with
# the CUDA build of PyTorch matching the AMI's toolkit, gsplat from git (PyPI
# is stale), psplat in editable mode, splat-transform, a checkout of gsplat's
# examples for training, and a systemd timer that stops the instance after
# 45 idle minutes. Writes the resulting versions to ~/psplat-bootstrap.json.
#
#   PSPLAT_GSPLAT_REF    gsplat git ref to install (default main; pin a SHA)
#   PSPLAT_TORCH_INDEX   override the PyTorch wheel index, e.g. .../whl/cu128
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$HOME/.venvs/psplat"
gsplat_ref="${PSPLAT_GSPLAT_REF:-main}"
log() { printf '[psplat] %s\n' "$*" >&2; }

log "system packages"
sudo dnf -y -q install git gcc gcc-c++ cmake ninja-build nodejs20 npm jq
# ffmpeg is not in the Amazon Linux 2023 repositories; renders for the video
# can be encoded on the Mac instead.
sudo dnf -y -q install ffmpeg-free 2>/dev/null || log "ffmpeg not available from dnf; skipped"

if ! command -v uv >/dev/null 2>&1; then
  log "uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
[ -x "$venv/bin/python" ] || uv venv -p 3.12 "$venv"

# Match PyTorch's CUDA build to the toolkit the AMI ships.
if [ -z "${PSPLAT_TORCH_INDEX:-}" ]; then
  cuda="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')"
  [ -n "$cuda" ] || cuda="$(jq -r '.cuda.version' /usr/local/cuda/version.json 2>/dev/null | awk -F. '{print $1 $2}')"
  [ -n "$cuda" ] || { log "cannot find the CUDA toolkit version; set PSPLAT_TORCH_INDEX"; exit 1; }
  PSPLAT_TORCH_INDEX="https://download.pytorch.org/whl/cu$cuda"
fi
log "PyTorch from $PSPLAT_TORCH_INDEX"
uv pip install --python "$venv" torch torchvision --index-url "$PSPLAT_TORCH_INDEX"

log "gsplat @ $gsplat_ref"
uv pip install --python "$venv" "git+https://github.com/nerfstudio-project/gsplat.git@$gsplat_ref"
# gsplat's training script and dataset parser live in its examples folder,
# which the wheel does not include; psplat bench train drives them.
if [ ! -d "$HOME/gsplat/.git" ]; then
  git clone -q https://github.com/nerfstudio-project/gsplat.git "$HOME/gsplat"
fi
git -C "$HOME/gsplat" fetch -q origin && git -C "$HOME/gsplat" checkout -q "$gsplat_ref"
uv pip install --python "$venv" -r "$HOME/gsplat/examples/requirements.txt" || \
  log "some gsplat example requirements failed to install; bench train may need them"

log "psplat"
uv pip install --python "$venv" -e "${repo}[dev,aws]"

log "splat-transform"
sudo npm install -g --silent @playcanvas/splat-transform@3.4.2

log "idle stop timer"
sudo install -m 0755 "$repo/scripts/aws_stop_idle.sh" /usr/local/bin/psplat-stop-idle
sudo tee /etc/systemd/system/psplat-stop-idle.service >/dev/null <<'EOF'
[Unit]
Description=Stop this instance after 45 idle minutes (psplat)

[Service]
Type=oneshot
ExecStart=/usr/local/bin/psplat-stop-idle
EOF
sudo tee /etc/systemd/system/psplat-stop-idle.timer >/dev/null <<'EOF'
[Unit]
Description=Check every 5 minutes whether this instance is idle (psplat)

[Timer]
OnBootSec=10min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now psplat-stop-idle.timer

log "checking"
nvidia-smi
"$venv/bin/python" - <<'EOF' | tee "$HOME/psplat-bootstrap.json"
import json, subprocess, torch, gsplat, psplat
print(json.dumps({
    "psplat": psplat.__version__,
    "gsplat": gsplat.__version__,
    "gsplat_ref": subprocess.run(["git", "-C", __import__("os").path.expanduser("~/gsplat"), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
EOF
log "done. Record these versions in docs/env.md. Activate with: source $venv/bin/activate"
