#!/bin/bash

# 设置变量
SCRIPT_DIR=$(dirname "$0")
ENV_PATH="$SCRIPT_DIR/env"

# 检测操作系统和架构
OS=$(uname -s)
ARCH=$(uname -m)

# 根据操作系统和架构设置 Miniconda 下载链接
if [ "$OS" == "Linux" ]; then
    if [ "$ARCH" == "x86_64" ]; then
        MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_24.9.2-0-Linux-x86_64.sh"
    elif [ "$ARCH" == "aarch64" ]; then
        MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_24.9.2-0-Linux-aarch64.sh"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
elif [ "$OS" == "Darwin" ]; then
    if [ "$ARCH" == "x86_64" ]; then
        MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_24.9.2-0-MacOSX-x86_64.sh"
    elif [ "$ARCH" == "arm64" ]; then
        MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_24.9.2-0-MacOSX-arm64.sh"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
else
    echo "Unsupported operating system: $OS"
    exit 1
fi

# 检查是否已安装 Conda
if command -v conda >/dev/null 2>&1; then
    echo "Conda is already installed."
else
    # 下载 Miniconda
    echo "Downloading Miniconda..."
    curl -L -o "miniconda.sh" "$MINICONDA_URL"

    # 安装 Miniconda
    echo "Installing Miniconda..."
    bash "miniconda.sh" -b -p "$HOME/miniconda3"

    # 添加 Miniconda 到 PATH (如果需要)
    if ! grep -q "$HOME/miniconda3/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
    fi

    # 删除安装脚本
    rm "miniconda.sh"
fi

# 创建虚拟环境
echo "Creating virtual environment..."
conda create -y -p "$ENV_PATH" python=3.12

# 激活环境并安装依赖
echo "Activating environment and installing dependencies..."
source activate "$ENV_PATH"
pip install -r requirements.txt

echo "Setup completed successfully!"