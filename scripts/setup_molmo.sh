#!/usr/bin/env bash
# Run on Alienware: bash scripts/setup_molmo.sh
# Installs Molmo in its own Conda environment; prints the hardware-run command.
set -Eeuo pipefail
trap 'printf "\nMolmo 安装失败（第 %s 行）。修复上面的错误后可重新执行此脚本。\n" "$LINENO" >&2' ERR

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        '用法：bash scripts/setup_molmo.sh' \
        '在 Alienware 创建或复用 molmo Conda 环境，安装依赖、下载并检查模型缓存。' \
        '已有 molmo 环境将安装为脚本指定的依赖版本；Python 必须为 3.11。' \
        '完成后打印折叠循环命令；本脚本不启动相机或机器人。'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf '不支持的参数；使用 --help 查看用法。\n' >&2
    exit 2
fi

molmo_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$molmo_project_root"

printf '\n[1/6] 检查 NVIDIA 驱动和 Conda\n'
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '找不到 nvidia-smi，请先在 Alienware 安装可用的 NVIDIA 驱动。\n' >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

molmo_conda_exe="${CONDA_EXE:-}"
if [[ -z "$molmo_conda_exe" || ! -x "$molmo_conda_exe" ]]; then
    molmo_conda_exe="$(type -P conda || true)"
fi
if [[ -z "$molmo_conda_exe" ]]; then
    for molmo_candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
        if [[ -x "$molmo_candidate" ]]; then
            molmo_conda_exe="$molmo_candidate"
            break
        fi
    done
fi
if [[ -z "$molmo_conda_exe" ]]; then
    printf '找不到 Conda。请安装 Miniconda，或设置 CONDA_EXE 为 conda 可执行文件路径。\n' >&2
    exit 1
fi
molmo_conda_base="$("$molmo_conda_exe" info --base)"
# Conda initialization scripts can refer to unset variables.
set +u
source "$molmo_conda_base/etc/profile.d/conda.sh"
set -u

printf '\n[2/6] 创建或复用 molmo 环境\n'
if conda run -n molmo python --version >/dev/null 2>&1; then
    printf '复用现有 molmo 环境，并安装以下指定版本。\n'
else
    conda create -n molmo python=3.11 -y
fi
set +u
conda activate molmo
set -u
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "已有 molmo 环境不是 Python 3.11，请先处理环境版本"; print("Molmo Python:", sys.executable)'
molmo_python_path="$(python -c 'import sys; print(sys.executable)')"

printf '\n[3/6] 安装 CUDA 12.8 版 PyTorch 和 Molmo 依赖\n'
python -m pip install --upgrade pip
python -m pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
    transformers==4.57.6 \
    accelerate==1.14.0 \
    bitsandbytes==0.50.2 \
    huggingface-hub==0.36.2 \
    einops==0.8.2 \
    sentencepiece==0.2.2 \
    safetensors==0.8.0 \
    numpy Pillow requests
python -m pip check

printf '\n[4/6] 检查 CUDA 和 GPU 运算\n'
python - <<'PY'
import torch
import transformers
import bitsandbytes

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
assert torch.cuda.is_available(), "CUDA 不可用，请检查 NVIDIA 驱动与 PyTorch runtime 的兼容性"
gpu = torch.cuda.get_device_properties(0)
print("GPU:", gpu.name)
print("显存:", round(gpu.total_memory / 1024**3, 1), "GiB")
probe = torch.ones((16, 16), device="cuda")
assert (probe @ probe).sum().item() == 4096
print("CUDA 运算通过")
PY

printf '\n[5/6] 下载 allenai/MolmoPoint-8B 到 Hugging Face 默认缓存\n'
printf '首次下载包含完整模型权重，耗时取决于网络；重新运行会复用已有缓存。\n'
hf download allenai/MolmoPoint-8B

printf '\n[6/6] 检查离线模型配置和处理器\n'
python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor

model = "allenai/MolmoPoint-8B"
cached = snapshot_download(model, local_files_only=True)
AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
print("模型缓存:", cached)
print("离线配置和处理器加载通过")
PY

printf '\nMolmo 安装和缓存检查完成；尚未执行完整模型推理。\n'
printf '独立 Python 路径：%s\n' "$molmo_python_path"
printf '请在原来的机器人 Python 环境中运行以下命令（会实际驱动 xArm）：\n\n'
printf 'cd %q\n' "$molmo_project_root"
printf 'python scripts/claude_fold_exploration.py \\\n'
printf '  --robot-config config/robot.example.json \\\n'
printf '  --perception-config config/perception.free_exploration.json \\\n'
printf '  --planner-backend remote \\\n'
printf '  --remote-planner-host company-planner \\\n'
printf '  --molmo-python %q \\\n' "$molmo_python_path"
printf '  --molmo-gpu-max-memory-gib 17 \\\n'
printf '  --max-iterations 0 \\\n'
printf '  --no-observer-camera \\\n'
printf '  --viser \\\n'
printf '  --real \\\n'
printf '  --confirm-real\n\n'
printf '上面的 17 GiB 模型放置预算沿用项目默认值，按 24 GB 级别 GPU 配置；不代表实际推理峰值。\n'
printf '首次使用远程桥接时，可先运行 scripts/remote_fold_smoke.py 验证已有 run。\n'
