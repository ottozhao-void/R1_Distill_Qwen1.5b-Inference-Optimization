# 新功能更新说明

本次更新根据 `patch_1.instructions.md` 实现了以下新功能：

## 🆕 新增功能

### 1. 量化推理模块
- 新增 `inference/inference_quantization.py` 模块
- 对比标准精度 vs 量化推理性能
- 支持8位和4位量化（使用bitsandbytes）
- 自动内存管理和清理

### 2. YAML配置系统
- 移除旧的 `ConfigPresets` 类
- 新增YAML配置文件支持
- 每个优化技术有独立的配置文件夹
- 支持三级批次配置：small, medium, large

### 3. 新性能指标
- **Time to First Token (TTFT)**: 首token生成时间
- **Time Per Output Token (TPOT)**: 每输出token平均时间
- 更精确的性能监控

### 4. Markdown报告格式
- 每个实验生成独立的MD报告
- 包含完整的配置参数和性能指标
- 自动生成对比分析报告

### 5. 优化技术选择
- 命令行参数支持选择优化技术
- 支持 PagedAttention 和 量化 两种技术
- 灵活的实验配置管理

## 📁 新目录结构

```
config/
├── PagedAttention/
│   ├── pagedattention_small_default.yaml
│   ├── pagedattention_medium_default.yaml
│   └── pagedattention_large_default.yaml
└── 量化/
    ├── 量化_small_default.yaml
    ├── 量化_medium_default.yaml
    └── 量化_large_default.yaml

results/
├── PagedAttention/
└── 量化/

inference/
├── inference_module.py (基类)
├── inference_paged_attention.py (PagedAttention实现)
└── inference_quantization.py (量化实现)
```

## 🚀 使用方法

### 基本使用

```bash
# PagedAttention测试 (默认medium批次)
python main.py --technique PagedAttention

# 量化测试 (small批次)
python main.py --technique 量化 --batch-level small

# 大批次测试
python main.py --technique PagedAttention --batch-level large
```

### 高级选项

```bash
# 使用自定义配置文件
python main.py --config config/PagedAttention/pagedattention_large_default.yaml

# 只运行优化推理
python main.py --technique 量化 --method optimized

# 内存基准测试
python main.py --technique PagedAttention --benchmark-memory

# 自定义实验名称
python main.py --technique 量化 --experiment-name accuracy_test --batch-level medium
```

### 配置文件示例

PagedAttention配置 (`config/PagedAttention/pagedattention_medium_default.yaml`):
```yaml
experiment_name: default
technique: PagedAttention
batch_level: medium
model: /data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B
device: [4]
max_tokens: 100
temperature: 0.8
top_p: 0.95
batch_size: 4
num_prompts: 10
warmup_iterations: 3
test_iterations: 5
save_results: true
quantization_config: null
```

量化配置 (`config/量化/量化_medium_default.yaml`):
```yaml
experiment_name: default
technique: 量化
batch_level: medium
# ... 其他参数相同 ...
quantization_config:
  load_in_8bit: true
  load_in_4bit: false
  bnb_4bit_compute_dtype: float16
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: nf4
```

## 📊 输出报告

每次实验会生成以下文件：

1. **基础推理报告**: `基础推理_report.md` / `基础推理_report.json`
2. **优化推理报告**: `优化推理_report.md` / `优化推理_report.json`  
3. **对比实验报告**: `comparison_experiment_{batch_level}_{experiment_name}.md`

报告包含：
- 完整的实验配置
- 详细的性能指标
- 改进分析和结论
- 原始数据记录

## 🔧 依赖更新

新增依赖：
```bash
pip install bitsandbytes>=0.41.0  # 量化支持
# pyyaml>=6.0 已在原requirements.txt中
```

## 🧪 测试新功能

运行测试脚本验证安装：
```bash
python test_new_features.py
```

## 📝 完整命令行参数

```
--technique {PagedAttention,量化}     选择优化技术
--batch-level {small,medium,large}    批次级别
--experiment-name EXPERIMENT_NAME     实验名称
--config CONFIG                       YAML配置文件路径
--method {basic,optimized,both}       推理方法
--benchmark-memory                    运行内存效率基准测试
--max-tokens MAX_TOKENS               最大生成tokens数量
--temperature TEMPERATURE             采样温度
--top-p TOP_P                         核采样参数
--batch-size BATCH_SIZE               批次大小
--num-prompts NUM_PROMPTS             测试提示数量
--test-iterations TEST_ITERATIONS     测试迭代次数
--warmup-iterations WARMUP_ITERATIONS 预热迭代次数
--output-dir OUTPUT_DIR               输出目录
--no-save                             不保存结果文件
--prompts PROMPTS [PROMPTS ...]       自定义测试提示
--prompts-file PROMPTS_FILE           从文件读取测试提示
```

## 📈 性能指标说明

新增的性能指标：

- **Time to First Token (TTFT)**: 从开始推理到生成第一个token的时间，重要的用户体验指标
- **Time Per Output Token (TPOT)**: 平均每个输出token的生成时间，反映推理效率
- **改进百分比**: 自动计算优化技术相对于基础推理的改进程度

这些指标帮助更全面地评估不同优化技术的效果。
