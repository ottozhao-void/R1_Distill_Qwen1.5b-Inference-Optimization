"""
推理模块基类 - InferenceModule

该模块定义了推理模块的基础接口，所有具体的推理模块都应该继承这个基类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from utils.performance_monitor import PerformanceMonitor, PerformanceMetrics
from config.config import Config


class InferenceModule(ABC):
    """推理模块基类"""
    
    def __init__(self, config: Config):
        """
        初始化推理模块
        
        Args:
            config: 配置对象
        """
        self.config = config
        # 从config中获取设备ID用于GPU内存监测
        device_id = None
        if config.device and len(config.device) > 0 and config.device[0] != 'cpu':
            device_id = config.device[0]
        self.performance_monitor = PerformanceMonitor(device_id=device_id)
        self.is_initialized = False
    
    @abstractmethod
    def basic_inference(self, prompts: List[str]) -> List[str]:
        """
        基础推理方法（对照组）
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            生成的文本列表
        """
        pass
    
    @abstractmethod
    def optimized_inference(self, prompts: List[str]) -> List[str]:
        """
        优化推理方法（实验组）
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            生成的文本列表
        """
        pass
    
    def inference(self, prompts: List[str], method: str = "both") -> Dict[str, Any]:
        """
        模型推理入口，调用基础推理和优化推理，并进行性能监测
        
        Args:
            prompts: 输入提示列表
            method: 推理方法 ("basic", "optimized", "both")
            
        Returns:
            包含结果和性能指标的字典
        """
        results = {}
        
        if method in ["basic", "both"]:
            print(f"\n开始基础推理测试...")
            basic_results, basic_metrics = self._run_inference_with_monitoring(
                prompts, self.basic_inference, "基础推理"
            )
            results["basic"] = {
                "outputs": basic_results,
                "metrics": basic_metrics
            }
        
        if method in ["optimized", "both"]:
            print(f"\n开始优化推理测试...")
            optimized_results, optimized_metrics = self._run_inference_with_monitoring(
                prompts, self.optimized_inference, "优化推理"
            )
            results["optimized"] = {
                "outputs": optimized_results,
                "metrics": optimized_metrics
            }
        
        # 如果两种方法都运行了，进行对比
        if method == "both":
            print(f"\n开始性能对比...")
            self.compare_results(results["basic"]["metrics"], results["optimized"]["metrics"])
        
        return results
    
    def _run_inference_with_monitoring(self, prompts: List[str], 
                                     inference_func, method_name: str) -> tuple:
        """
        运行推理并监测性能
        
        Args:
            prompts: 输入提示列表
            inference_func: 推理函数
            method_name: 方法名称
            
        Returns:
            (outputs, metrics) 元组
        """
        import torch
        
        # 重置监测器 - 清理之前的内存统计
        self.performance_monitor.reset()
        
        # 记录开始前的内存状态
        initial_gpu_info = self.performance_monitor.get_gpu_memory_info()
        print(f"开始前GPU内存: 已分配={initial_gpu_info['allocated']:.2f}MB, 峰值={initial_gpu_info['peak']:.2f}MB, 系统使用={initial_gpu_info['system_used']:.2f}MB")
        
        # 预热阶段
        print(f"预热阶段...")
        warmup_prompts = prompts[:min(len(prompts), self.config.warmup_iterations)]
        for _ in range(self.config.warmup_iterations):
            _ = inference_func(warmup_prompts)
        
        # 预热后记录内存并重置监测
        warmup_gpu_info = self.performance_monitor.get_gpu_memory_info()
        print(f"预热后GPU内存: 已分配={warmup_gpu_info['allocated']:.2f}MB, 峰值={warmup_gpu_info['peak']:.2f}MB, 系统使用={warmup_gpu_info['system_used']:.2f}MB")
        
        # 重置系统内存基线（以预热后的状态为基准）
        self.performance_monitor.system_memory_tracker.reset_baseline()
        self.performance_monitor.start_monitoring()
        
        # 正式测试
        print(f"正式测试阶段...")
        all_outputs = []
        
        for i in range(self.config.test_iterations):
            with self.performance_monitor.measure_batch():
                outputs = inference_func(prompts)
                all_outputs.extend(outputs)
                
                # 记录token数量
                total_tokens = sum(len(output.split()) for output in outputs)
                self.performance_monitor.add_token_count(total_tokens)
        
        self.performance_monitor.stop_monitoring()
        
        # 记录结束后的内存状态
        final_gpu_info = self.performance_monitor.get_gpu_memory_info()
        system_memory_increase = self.performance_monitor.system_memory_tracker.get_memory_increase()
        print(f"结束后GPU内存: 已分配={final_gpu_info['allocated']:.2f}MB, 峰值={final_gpu_info['peak']:.2f}MB, 系统使用={final_gpu_info['system_used']:.2f}MB, 系统增量={system_memory_increase:.2f}MB")
        
        # 获取性能指标
        metrics = self.performance_monitor.get_metrics(
            model_name=self.config.model,
            batch_size=len(prompts)
        )
        
        # 打印结果
        self.print_result(metrics, method_name)
        
        return all_outputs, metrics
    
    def print_result(self, metrics: PerformanceMetrics, method_name: str):
        """
        打印性能监测结果并保存报告
        
        Args:
            metrics: 性能指标
            method_name: 方法名称
        """
        print(f"\n{method_name} 性能结果:")
        self.performance_monitor.print_summary(metrics)
        
        # 保存报告
        if self.config.save_results:
            import os
            
            # 保存JSON格式详细报告
            json_filename = f"{method_name.replace(' ', '_').lower()}_report.json"
            json_filepath = os.path.join(self.config.output_dir, json_filename)
            self.performance_monitor.save_detailed_report(metrics, json_filepath)
            
            # 保存Markdown格式报告
            md_filename = f"{method_name.replace(' ', '_').lower()}_report.md"
            md_filepath = os.path.join(self.config.output_dir, md_filename)
            
            # 创建配置字典
            config_dict = {
                'experiment_name': getattr(self.config, 'experiment_name', 'default'),
                'technique': getattr(self.config, 'technique', 'Unknown'),
                'batch_level': getattr(self.config, 'batch_level', 'medium'),
                'model': self.config.model,
                'device': self.config.device,
                'batch_size': self.config.batch_size,
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'num_prompts': self.config.num_prompts,
                'warmup_iterations': self.config.warmup_iterations,
                'test_iterations': self.config.test_iterations,
                'quantization_config': getattr(self.config, 'quantization_config', None)
            }
            
            self.performance_monitor.save_markdown_report(
                metrics, config_dict, md_filepath, method_name
            )
    
    def compare_results(self, basic_metrics: PerformanceMetrics, 
                       optimized_metrics: PerformanceMetrics):
        """
        比较基础推理和优化推理的结果
        
        Args:
            basic_metrics: 基础推理性能指标
            optimized_metrics: 优化推理性能指标
        """
        from utils.performance_monitor import compare_metrics
        compare_metrics(basic_metrics, optimized_metrics, "基础推理", "优化推理")
        
        # 保存比较报告
        if self.config.save_results:
            self._save_comparison_report(basic_metrics, optimized_metrics)
    
    def _save_comparison_report(self, basic_metrics: PerformanceMetrics, 
                              optimized_metrics: PerformanceMetrics):
        """保存对比实验的综合报告"""
        import os
        from datetime import datetime
        
        # 创建配置字典
        config_dict = {
            'experiment_name': getattr(self.config, 'experiment_name', 'default'),
            'technique': getattr(self.config, 'technique', 'Unknown'),
            'batch_level': getattr(self.config, 'batch_level', 'medium'),
            'model': self.config.model,
            'device': self.config.device,
            'batch_size': self.config.batch_size,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'num_prompts': self.config.num_prompts,
            'warmup_iterations': self.config.warmup_iterations,
            'test_iterations': self.config.test_iterations,
            'quantization_config': getattr(self.config, 'quantization_config', None)
        }
        
        # 计算改进指标
        latency_improvement = 0
        throughput_improvement = 0 
        memory_improvement = 0
        ttft_improvement = 0
        tpot_improvement = 0
        
        if basic_metrics.avg_latency > 0:
            latency_improvement = (basic_metrics.avg_latency - optimized_metrics.avg_latency) / basic_metrics.avg_latency * 100
        
        if basic_metrics.tokens_per_second > 0:
            throughput_improvement = (optimized_metrics.tokens_per_second - basic_metrics.tokens_per_second) / basic_metrics.tokens_per_second * 100
        
        if basic_metrics.gpu_memory_peak > 0:
            memory_improvement = (basic_metrics.gpu_memory_peak - optimized_metrics.gpu_memory_peak) / basic_metrics.gpu_memory_peak * 100
        
        if basic_metrics.time_to_first_token > 0:
            ttft_improvement = (basic_metrics.time_to_first_token - optimized_metrics.time_to_first_token) / basic_metrics.time_to_first_token * 100
        
        if basic_metrics.time_per_output_token > 0:
            tpot_improvement = (basic_metrics.time_per_output_token - optimized_metrics.time_per_output_token) / basic_metrics.time_per_output_token * 100
        
        report_md = f"""# {config_dict['technique']} 对比实验完整报告

## 实验概述
**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验名称**: {config_dict['experiment_name']}  
**优化技术**: {config_dict['technique']}  
**批次级别**: {config_dict['batch_level']}  

## 实验配置

### 模型配置
- **模型**: {config_dict['model']}
- **设备**: {config_dict['device']}
- **批次大小**: {config_dict['batch_size']}

### 推理参数
- **最大tokens**: {config_dict['max_tokens']}
- **温度**: {config_dict['temperature']}
- **Top-p**: {config_dict['top_p']}

### 测试参数
- **提示数量**: {config_dict['num_prompts']}
- **预热迭代**: {config_dict['warmup_iterations']}
- **测试迭代**: {config_dict['test_iterations']}

"""
        
        if config_dict.get('quantization_config'):
            import json
            report_md += f"""### 量化配置
```yaml
{json.dumps(config_dict['quantization_config'], indent=2)}
```

"""
        
        report_md += f"""## 基础推理实验结果

### 延迟指标
- **总时间**: {basic_metrics.total_time:.3f}s
- **平均延迟**: {basic_metrics.avg_latency*1000:.2f}ms
- **P50延迟**: {basic_metrics.p50_latency*1000:.2f}ms
- **P95延迟**: {basic_metrics.p95_latency*1000:.2f}ms
- **P99延迟**: {basic_metrics.p99_latency*1000:.2f}ms
- **首token时间**: {basic_metrics.time_to_first_token*1000:.2f}ms
- **每输出token时间**: {basic_metrics.time_per_output_token*1000:.2f}ms

### 吞吐量指标
- **Tokens/秒**: {basic_metrics.tokens_per_second:.2f}
- **请求/秒**: {basic_metrics.requests_per_second:.2f}

### 内存使用
- **GPU内存峰值**: {basic_metrics.gpu_memory_peak:.2f}MB
- **GPU内存当前**: {basic_metrics.gpu_memory_used:.2f}MB
- **CPU内存**: {basic_metrics.cpu_memory_used:.2f}MB

## 优化推理实验结果

### 延迟指标
- **总时间**: {optimized_metrics.total_time:.3f}s
- **平均延迟**: {optimized_metrics.avg_latency*1000:.2f}ms
- **P50延迟**: {optimized_metrics.p50_latency*1000:.2f}ms
- **P95延迟**: {optimized_metrics.p95_latency*1000:.2f}ms
- **P99延迟**: {optimized_metrics.p99_latency*1000:.2f}ms
- **首token时间**: {optimized_metrics.time_to_first_token*1000:.2f}ms
- **每输出token时间**: {optimized_metrics.time_per_output_token*1000:.2f}ms

### 吞吐量指标
- **Tokens/秒**: {optimized_metrics.tokens_per_second:.2f}
- **请求/秒**: {optimized_metrics.requests_per_second:.2f}

### 内存使用
- **GPU内存峰值**: {optimized_metrics.gpu_memory_peak:.2f}MB
- **GPU内存当前**: {optimized_metrics.gpu_memory_used:.2f}MB
- **CPU内存**: {optimized_metrics.cpu_memory_used:.2f}MB

## 性能改进分析

### 改进指标对比
| 指标 | 基础推理 | 优化推理 | 改进 |
|------|----------|----------|------|
| 平均延迟 | {basic_metrics.avg_latency*1000:.2f}ms | {optimized_metrics.avg_latency*1000:.2f}ms | {latency_improvement:+.1f}% |
| 吞吐量 | {basic_metrics.tokens_per_second:.2f} tokens/s | {optimized_metrics.tokens_per_second:.2f} tokens/s | {throughput_improvement:+.1f}% |
| GPU内存峰值 | {basic_metrics.gpu_memory_peak:.2f}MB | {optimized_metrics.gpu_memory_peak:.2f}MB | {memory_improvement:+.1f}% |
| 首token时间 | {basic_metrics.time_to_first_token*1000:.2f}ms | {optimized_metrics.time_to_first_token*1000:.2f}ms | {ttft_improvement:+.1f}% |
| 每输出token时间 | {basic_metrics.time_per_output_token*1000:.2f}ms | {optimized_metrics.time_per_output_token*1000:.2f}ms | {tpot_improvement:+.1f}% |

### 优化效果总结
"""
        
        if throughput_improvement > 10:
            report_md += f"- ✅ **显著性能提升**: {config_dict['technique']}技术显著提升了推理性能\n"
        elif throughput_improvement > 0:
            report_md += f"- ✅ **适度性能提升**: {config_dict['technique']}技术适度提升了推理性能\n"
        else:
            report_md += f"- ⚠️ **性能无明显提升**: 在当前测试场景下，{config_dict['technique']}优化效果不明显\n"
        
        if memory_improvement > 10:
            report_md += f"- 💾 **显著内存节省**: 内存使用减少了{memory_improvement:.1f}%\n"
        elif memory_improvement > 0:
            report_md += f"- 💾 **适度内存节省**: 内存使用减少了{memory_improvement:.1f}%\n"
        
        if latency_improvement > 10:
            report_md += f"- 🚀 **显著延迟降低**: 平均延迟降低了{latency_improvement:.1f}%\n"
        elif latency_improvement > 0:
            report_md += f"- 🚀 **适度延迟降低**: 平均延迟降低了{latency_improvement:.1f}%\n"
        
        report_md += f"""
## 结论
本次实验对比了基础推理与使用{config_dict['technique']}技术的优化推理性能。"""
        
        if throughput_improvement > 0 and memory_improvement > 0:
            report_md += f"结果显示{config_dict['technique']}技术在提升推理速度和节省内存方面都有良好表现。"
        elif throughput_improvement > 0:
            report_md += f"结果显示{config_dict['technique']}技术主要在提升推理速度方面有良好表现。"
        elif memory_improvement > 0:
            report_md += f"结果显示{config_dict['technique']}技术主要在节省内存使用方面有良好表现。"
        else:
            report_md += f"在当前测试配置下，{config_dict['technique']}技术的优化效果有限，可能需要调整配置参数或在更大规模的测试中验证效果。"
        
        report_md += f"""

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存完整对比报告
        comparison_filename = f"comparison_experiment_{config_dict['batch_level']}_{config_dict['experiment_name']}.md"
        comparison_filepath = os.path.join(self.config.output_dir, comparison_filename)
        
        try:
            with open(comparison_filepath, 'w', encoding='utf-8') as f:
                f.write(report_md)
            print(f"完整对比实验报告已保存到: {comparison_filepath}")
        except Exception as e:
            print(f"保存对比实验报告失败: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_name": self.config.model,
            "device": self.config.get_device_str(),
            "tensor_parallel_size": self.config.get_tensor_parallel_size(),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
    
    def generate_test_prompts(self) -> List[str]:
        """
        生成测试用的提示列表
        
        Returns:
            测试提示列表
        """
        base_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important discovery in science was",
            "Climate change affects our planet by",
            "The key to successful leadership involves",
            "Machine learning algorithms can help us",
            "The evolution of human civilization shows",
            "Renewable energy sources are important because",
            "The impact of social media on society",
            "Space exploration has taught us that",
            "The benefits of education include",
            "Sustainable development requires",
            "The role of government in modern society",
            "Innovation drives progress through",
            "The importance of healthcare systems"
        ]
        
        # 根据配置中的num_prompts选择提示
        selected_prompts = base_prompts[:min(len(base_prompts), self.config.num_prompts)]
        return selected_prompts
    
    @abstractmethod
    def cleanup(self):
        """
        清理资源
        """
        pass
