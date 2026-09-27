"""P0.10: the offline-testable parts of the AWS scripts.

Provisioning itself needs an AWS account and is run by hand from the Mac.
What can be checked here is the AMI parameter choice, which must pick
exactly one Deep Learning Base GPU AMI for Amazon Linux 2023 and refuse
to guess otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "scripts" / "aws_lib.sh"
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the AWS scripts are operator tools run from macOS or Linux",
)
P = "/aws/service/deeplearning/ami/x86_64"


def pick(names: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{LIB}"; pick_dlami_param'],
        input="\n".join(names) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )


def test_picks_the_single_base_al2023_parameter() -> None:
    names = [
        f"{P}/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id",
        f"{P}/base-oss-nvidia-driver-gpu-amazon-linux-2023/2026.09.01/ami-id",
        f"{P}/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
        f"{P}/oss-nvidia-driver-gpu-pytorch-2.7-amazon-linux-2023/latest/ami-id",
        f"{P}/base-neuron-amazon-linux-2023/latest/ami-id",
    ]
    result = pick(names)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == names[0]


def test_refuses_to_guess_between_several() -> None:
    result = pick(
        [
            f"{P}/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id",
            f"{P}/base-multi-cuda-gpu-amazon-linux-2023/latest/ami-id",
        ]
    )
    assert result.returncode == 1
    assert "PSPLAT_AMI_SSM_PARAM" in result.stderr and "multi-cuda" in result.stderr


def test_reports_when_nothing_matches() -> None:
    result = pick([f"{P}/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"])
    assert result.returncode == 1 and "no Deep Learning Base AMI" in result.stderr
