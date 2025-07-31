# DeepSeek-R1 大语言模型推理优化项目

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![vLLM](https://img.shields.io/badge/vLLM-0.2.5%2B-purple.svg)

🚀 一个专为DeepSeek-R1-Distill-Qwen-1.5B模型设计的推理优化和性能测试框架，支持多种优化技术的对比分析。

## 📋 目录

- [项目概述](#项目概述)
- [特性](#特性)
- [快速开始](#快速开始)
- [安装](#安装)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [贡献指南](#贡献指南)

## 🎯 项目概述

本项目专注于大语言模型推理性能优化，通过对比传统推理方法与现代优化技术，为模型部署提供性能基准和优化建议。目前支持以下优化技术：

- **PagedAttention**: 内存高效的注意力机制
- **模型量化**: 降低模型精度以提升推理速度

## ✨ 特性

### 🔧 核心功能
- 多种推理优化技术支持（PagedAttention、量化等）
- 自动化性能对比测试
- 详细的性能指标分析（延迟、吞吐量、内存使用）
- 可配置的批次大小和测试参数
- 支持多GPU并行推理

### 📊 性能监控
- 实时GPU/CPU内存监控
- 详细的延迟分析（P50、P95、P99）
- 吞吐量统计
- 自动生成性能报告

### 🛠️ 易用性
- YAML配置文件支持
- 命令行界面
- 自动化实验结果保存
- 灵活的测试提示配置

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)
- 8GB+ GPU内存 (推荐)

### 安装

1. **克隆仓库**
```bash
git clone https://github.com/ottozhao-void/R1_Distill_Qwen1.5b-Inference-Optimization.git
cd R1_Distill_Qwen1.5b-Inference-Optimization
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备模型**
确保DeepSeek-R1-Distill-Qwen-1.5B模型已下载并配置正确路径。

### 基本使用

**运行PagedAttention对比测试：**
```bash
python main.py --technique PagedAttention --batch-level medium
```

**运行量化对比测试：**
```bash
python main.py --technique 量化 --batch-level small
```

**查看可用设备：**
```bash
python main.py --list-devices
```

## 📖 使用指南

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--technique` | 优化技术 (PagedAttention/量化) | PagedAttention |
| `--batch-level` | 批次级别 (small/medium/large) | medium |
| `--experiment-name` | 实验名称 | default |
| `--device` | 指定GPU设备 (如: 0,1) | 自动检测 |
| `--max-tokens` | 最大生成tokens | 100 |
| `--batch-size` | 批次大小 | 4 |
| `--method` | 推理方法 (basic/optimized/both) | both |

### 配置文件使用

使用自定义配置文件：
```bash
python main.py --config config/PagedAttention/custom_config.yaml
```

### 内存效率测试

运行内存基准测试：
```bash
python main.py --benchmark-memory --batch-sizes "1,2,4,8,16"
```

### 自定义测试提示

使用自定义提示：
```bash
python main.py --prompts "解释人工智能" "写一首诗" --technique PagedAttention
```

从文件加载提示：
```bash
python main.py --prompts-file inference/example_prompts.txt
```

## ⚙️ 配置说明

### 配置文件结构

```yaml
# 实验配置
experiment_name: "default"
technique: "PagedAttention"
batch_level: "medium"

# 模型配置
model: "/path/to/DeepSeek-R1-Distill-Qwen-1.5B"
device: [0, 1]  # GPU设备ID

# 推理参数
max_tokens: 100
temperature: 0.8
top_p: 0.95
batch_size: 4

# 测试参数
num_prompts: 10
warmup_iterations: 3
test_iterations: 5

# 输出配置
output_dir: "results"
save_results: true
```

### 批次级别说明

- **small**: 轻量级测试 (batch_size≤2, num_prompts≤5)
- **medium**: 中等规模测试 (默认配置)
- **large**: 大规模测试 (batch_size≥8, num_prompts≥20)

## 📊 实验结果

### 性能改进示例

基于PagedAttention技术的测试结果：

| 指标 | 基础推理 | 优化推理 | 改进 |
|------|----------|----------|------|
| 平均延迟 | 6,671ms | 1,119ms | +83.2% |
| 吞吐量 | 30.92 tokens/s | 189.32 tokens/s | +512.2% |

### 输出文件

每次实验会生成以下文件：
- `基础推理_report.json/md`: 基础推理性能报告
- `优化推理_report.json/md`: 优化推理性能报告  
- `comparison_experiment_{config}.md`: 完整对比分析报告

## 📁 项目结构

```
R1_Distill_Qwen1.5b-Inference-Optimization/
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖列表
├── config/                 # 配置文件目录
│   ├── config.py          # 配置管理模块
│   ├── PagedAttention/    # PagedAttention配置
│   └── 量化/               # 量化配置
├── inference/             # 推理模块
│   ├── inference_module.py           # 基础推理模块
│   ├── inference_paged_attention.py  # PagedAttention推理
│   ├── inference_quantization.py     # 量化推理
│   └── example_prompts.txt          # 示例提示
├── utils/                 # 工具模块
│   └── performance_monitor.py       # 性能监控
└── results/              # 实验结果
    ├── PagedAttention/   # PagedAttention实验结果
    └── 量化/              # 量化实验结果
```

## 🔧 开发

### 添加新的优化技术

1. 在`inference/`目录下创建新的推理模块
2. 继承`InferenceModule`基类
3. 实现必要的接口方法
4. 在`main.py`中注册新技术

### 自定义性能指标

编辑`utils/performance_monitor.py`添加新的监控指标。
