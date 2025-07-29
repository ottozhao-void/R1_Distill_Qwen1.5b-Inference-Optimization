#!/bin/bash

# LLM推理性能测试脚本
# 用于自动化运行多种配置的性能测试

echo "🚀 开始LLM推理性能自动化测试"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请确保Python已安装"
    exit 1
fi

# 检查依赖包
echo "📦 检查依赖包..."
python -c "import torch, transformers, vllm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少必要的依赖包，请运行: pip install -r requirements.txt"
    exit 1
fi

echo "✅ 依赖包检查通过"

# 创建结果目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="benchmark_results_$TIMESTAMP"
mkdir -p $RESULTS_DIR

echo "📁 结果将保存到: $RESULTS_DIR"

# 测试配置
MODELS=("/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B")
BATCH_SIZES=(1 2 4)
MAX_TOKENS=(50 100)

echo "🧪 开始基准测试..."

# 小模型快速测试
echo ""
echo "=== 小模型快速测试 ==="
python main.py --preset small --output-dir "$RESULTS_DIR/small_model" --test-iterations 3

# 不同批次大小测试
echo ""
echo "=== 内存效率基准测试 ==="
python main.py --preset small --benchmark-memory --batch-sizes 1,2,4,8 --output-dir "$RESULTS_DIR/memory_benchmark"

# 详细性能测试
for model in "${MODELS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for max_tokens in "${MAX_TOKENS[@]}"; do
            echo ""
            echo "=== 测试配置: $model, batch=$batch_size, tokens=$max_tokens ==="
            
            output_subdir="$RESULTS_DIR/detailed_$(basename $model)_b${batch_size}_t${max_tokens}"
            
            python main.py \
                --model "$model" \
                --batch-size "$batch_size" \
                --max-tokens "$max_tokens" \
                --num-prompts 5 \
                --test-iterations 3 \
                --output-dir "$output_subdir" \
                --method both
            
            if [ $? -ne 0 ]; then
                echo "⚠️ 测试失败: $model (batch=$batch_size, tokens=$max_tokens)"
            else
                echo "✅ 测试完成: $model (batch=$batch_size, tokens=$max_tokens)"
            fi
        done
    done
done

# 生成测试总结
echo ""
echo "📊 生成测试总结..."
cat > "$RESULTS_DIR/test_summary.md" << EOF
# LLM推理性能测试总结

## 测试时间
- 开始时间: $(date)
- 结果目录: $RESULTS_DIR

## 测试配置
- 模型列表: ${MODELS[@]}
- 批次大小: ${BATCH_SIZES[@]}
- 最大tokens: ${MAX_TOKENS[@]}

## 测试项目
1. 小模型快速测试
2. 内存效率基准测试
3. 详细性能对比测试

## 结果文件
- small_model/: 小模型测试结果
- memory_benchmark/: 内存基准测试结果
- detailed_*/: 详细测试结果

## 分析建议
1. 查看各配置下的吞吐量和延迟对比
2. 关注GPU内存使用效率
3. 分析PagedAttention的优化效果

EOF

echo ""
echo "🎉 所有测试完成！"
echo "📋 测试总结已保存到: $RESULTS_DIR/test_summary.md"
echo "📊 详细结果请查看: $RESULTS_DIR/"

# 显示关键统计信息
echo ""
echo "📈 关键统计信息:"
echo "----------------"
find "$RESULTS_DIR" -name "*.json" -type f | wc -l | xargs echo "生成报告文件数:"
find "$RESULTS_DIR" -type d | wc -l | xargs echo "测试配置数:"

echo ""
echo "🔍 查看结果建议:"
echo "  1. 浏览各子目录中的JSON报告文件"
echo "  2. 对比basic和optimized的性能指标"
echo "  3. 分析不同配置下的性能变化趋势"
