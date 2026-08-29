# NV Fleet

同时看多台远程/本机服务器的 GPU 状态。类似 [nvitop](https://github.com/XuehaiPan/nvitop)，但是跨机器的。

## 它能干什么

左边勾选服务器，右边实时显示每块 GPU 的利用率、显存、温度、功率，**ENC/DEC（NVENC 编码 / NVDEC 解码）占用与会话数**。

每块 GPU 可以展开看全量进程列表：自己的进程完整显示（PID/用户/显存/SM%、MEM%、ENC%、DEC%/命令行，绿色高亮），别人的进程聚合显示（PID/名称/显存，灰色；SM% 受驱动 cgroup 隔离限制可能不可见）。

远程服务器不需要装任何东西，有 NVIDIA 驱动和 Python 3 就行。数据全部通过 SSH 传输，不写文件。本机数据不走 SSH（transport=local），直接底层执行同一份探针。

## 安装（Anaconda）

```bash
conda env create -f environment.yml
conda activate nvfleet
pip install -e .
```

本地需要 Python 3.10+。远程服务器需要 NVIDIA 驱动和 Python 3，配置好 SSH 免密登录。

## 使用

```bash
./nvfleet        # 一键启动（自动激活 conda 环境）
```

或者手动 `conda activate nvfleet && nvfleet`。

| 按键 | 作用 |
|------|------|
| `↑` `↓` | 左侧服务器列表/面板内 GPU 行移动 |
| `Space` | 服务器列表：勾选/取消；面板内：展开/收起当前 GPU 进程 |
| `Enter` | 在面板内展开/收起当前 GPU 进程列表 |
| `r` | 强制刷新所有已选服务器 |
| `c` | 紧凑模式 |
| `q` | 退出 |

## 配置

启动时自动读 `~/.ssh/config`，把里面的 Host 都列出来（github.com 这类代码托管域名自动跳过）。本机作为 `local` 项默认显示。

如果想自定义显示名或默认勾选，可以建 `~/.config/nvfleet/servers.yml`：

```yaml
local: true              # 包含本机（未配置 YAML 时默认包含）
refresh_seconds: 1.5
timeout_seconds: 5.0
servers:
  - host: two4090
    label: "2x RTX 4090"
    enabled: true
  - host: a100-server
    label: "8x A100"
```

## 原理

本地通过 SSH 把一段 Python 脚本发送到远程服务器执行（`ssh host "python3 -"`，脚本从 stdin 读取），本机模式直接用本地解释器 stdin 执行同一脚本。脚本用 ctypes 直接调 NVIDIA 驱动的 C 库（`libnvidia-ml.so`）拿 GPU 数据，和 nvitop 一样的方式。结果以 JSON 从 stdout 返回，本地解析后渲染。

GPU 进程信息同时查询 NVML 的 compute 和 graphics 列表（Xorg/FFmpeg NVENC 这类图形上下文也可见），每进程利用率取自驱动的采样缓冲（该功能需要驱动 ≥460；无样本时显示 `-`）。NVML 权限不足时自动降级 `nvidia-smi`。

整个过程不创建任何临时文件。

## License

MIT
