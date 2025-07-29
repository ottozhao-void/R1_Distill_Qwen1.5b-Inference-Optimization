# LLM 推理优化配置

## 模型配置

本项目已统一配置为使用 **DeepSeek-R1-Distill-Qwen-1.5B** 模型，确保整个推理工作流的一致性。

### 配置详情

- **默认模型路径**: `/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B`
- **所有预设配置**: 已更新为使用相同模型
- **示例代码**: 已更新为使用相同模型
- **基准测试脚本**: 已更新为使用相同模型

### 配置文件更新

1. **`config.py`**: 
   - 默认配置使用 DeepSeek-R1-Distill-Qwen-1.5B
   - 所有预设配置 (`small_model_config`, `large_model_config`, `llama_config`) 都使用同一模型
   - 修复了数据类的可变默认值问题

2. **`example.py`**: 
   - 自定义配置示例更新为使用 DeepSeek 模型

3. **`run_benchmark.sh`**: 
   - 基准测试脚本已配置为使用正确的模型路径

### 验证配置

运行配置验证脚本来确认所有配置都正确：

```bash
# 验证配置
python validate_config.py

# 或者使其可执行并运行
./validate_config.py
```

### 快速开始

配置完成后，可以运行以下命令开始测试：

```bash
# 小模型快速测试
python main.py --preset small

# 运行示例
python example.py

# 完整基准测试
./run_benchmark.sh
```

### 配置要点

- ✅ 统一模型路径，确保对比实验的公平性
- ✅ 保持相同的超参数设置
- ✅ 使用相同的设备配置
- ✅ 验证模型文件完整性
- ✅ 检查依赖包版本

这样的配置确保了 PagedAttention 与传统 Key-Value Cache 的对比实验中，唯一的变量就是内存管理机制的差异。
