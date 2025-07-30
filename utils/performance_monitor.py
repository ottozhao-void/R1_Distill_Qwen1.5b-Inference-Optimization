"""
性能检测模块 - PerformanceMonitor

该模块用于监测模型推理过程中的各项性能指标，包括吞吐量、延迟等。
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    
    # 延迟相关指标 (秒)
    total_time: float = 0.0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    
    # 新增性能指标
    time_to_first_token: float = 0.0  # 首token生成时间
    time_per_output_token: float = 0.0  # 每输出token平均时间
    
    # 吞吐量指标
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # 内存使用 (MB)
    gpu_memory_used: float = 0.0
    gpu_memory_peak: float = 0.0
    cpu_memory_used: float = 0.0
    
    # 模型和硬件信息
    model_name: str = ""
    device_info: str = ""
    batch_size: int = 0
    sequence_length: int = 0
    
    # 原始数据
    latencies: List[float] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)
    first_token_times: List[float] = field(default_factory=list)  # 首token时间列表
    
    def calculate_derived_metrics(self):
        """计算派生指标"""
        if self.latencies:
            self.avg_latency = float(np.mean(self.latencies))
            self.p50_latency = float(np.percentile(self.latencies, 50))
            self.p95_latency = float(np.percentile(self.latencies, 95))
            self.p99_latency = float(np.percentile(self.latencies, 99))
            
            total_tokens = sum(self.token_counts)
            if self.total_time > 0:
                self.tokens_per_second = total_tokens / self.total_time
                self.requests_per_second = len(self.latencies) / self.total_time
                
        # 计算首token时间
        if self.first_token_times:
            self.time_to_first_token = float(np.mean(self.first_token_times))
            
        # 计算每输出token时间
        if self.token_counts and self.latencies:
            total_output_tokens = sum(self.token_counts)
            total_generation_time = sum(self.latencies)
            if total_output_tokens > 0:
                self.time_per_output_token = total_generation_time / total_output_tokens


class PerformanceMonitor:
    """性能监测器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置监测器"""
        self.start_time = None
        self.end_time = None
        self.latencies = []
        self.token_counts = []
        self.first_token_times = []
        self.gpu_memory_peak = 0.0
        self.current_batch_start = None
        
        # GPU内存监测
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    @contextmanager
    def measure_batch(self):
        """测量单个批次的性能"""
        self.current_batch_start = time.time()
        
        # 记录开始时的GPU内存
        gpu_memory_start = 0.0
        if torch.cuda.is_available():
            gpu_memory_start = torch.cuda.memory_allocated() / 1024**2
        
        try:
            yield
        finally:
            # 记录结束时间和内存
            batch_time = time.time() - self.current_batch_start
            self.latencies.append(batch_time)
            
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**2
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                self.gpu_memory_peak = max(self.gpu_memory_peak, peak_memory)
    
    @contextmanager 
    def measure_first_token(self):
        """测量首token生成时间"""
        start_time = time.time()
        try:
            yield
        finally:
            first_token_time = time.time() - start_time
            self.first_token_times.append(first_token_time)
    
    def start_monitoring(self):
        """开始监测"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def stop_monitoring(self):
        """停止监测"""
        self.end_time = time.time()
    
    def add_token_count(self, count: int):
        """添加token数量"""
        self.token_counts.append(count)
    
    def get_metrics(self, model_name: str = "", batch_size: int = 0) -> PerformanceMetrics:
        """获取性能指标"""
        metrics = PerformanceMetrics()
        
        # 基本信息
        metrics.model_name = model_name
        metrics.batch_size = batch_size
        metrics.device_info = self._get_device_info()
        
        # 时间相关
        if self.start_time and self.end_time:
            metrics.total_time = self.end_time - self.start_time
        
        # 原始数据
        metrics.latencies = self.latencies.copy()
        metrics.token_counts = self.token_counts.copy()
        metrics.first_token_times = self.first_token_times.copy()
        
        # 内存使用
        metrics.gpu_memory_peak = self.gpu_memory_peak
        metrics.gpu_memory_used = self._get_current_gpu_memory()
        metrics.cpu_memory_used = self._get_cpu_memory()
        
        # 计算派生指标
        metrics.calculate_derived_metrics()
        
        return metrics
    
    def _get_device_info(self) -> str:
        """获取设备信息"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            return f"GPU: {device_name} (x{device_count})"
        else:
            return "CPU"
    
    def _get_current_gpu_memory(self) -> float:
        """获取当前GPU内存使用 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    
    def _get_cpu_memory(self) -> float:
        """获取CPU内存使用 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024**2
    
    def print_summary(self, metrics: PerformanceMetrics):
        """打印性能摘要"""
        print("\n" + "="*60)
        print("性能测试摘要")
        print("="*60)
        print(f"模型: {metrics.model_name}")
        print(f"设备: {metrics.device_info}")
        print(f"批次大小: {metrics.batch_size}")
        print(f"总请求数: {len(metrics.latencies)}")
        print(f"总时间: {metrics.total_time:.2f}s")
        print()
        
        print("延迟指标:")
        print(f"  平均延迟: {metrics.avg_latency*1000:.2f}ms")
        print(f"  P50延迟: {metrics.p50_latency*1000:.2f}ms")
        print(f"  P95延迟: {metrics.p95_latency*1000:.2f}ms")
        print(f"  P99延迟: {metrics.p99_latency*1000:.2f}ms")
        print(f"  首token时间: {metrics.time_to_first_token*1000:.2f}ms")
        print(f"  每输出token时间: {metrics.time_per_output_token*1000:.2f}ms")
        print()
        
        print("吞吐量指标:")
        print(f"  Tokens/秒: {metrics.tokens_per_second:.2f}")
        print(f"  请求/秒: {metrics.requests_per_second:.2f}")
        print()
        
        print("内存使用:")
        print(f"  GPU内存峰值: {metrics.gpu_memory_peak:.2f}MB")
        print(f"  GPU内存当前: {metrics.gpu_memory_used:.2f}MB")
        print(f"  CPU内存: {metrics.cpu_memory_used:.2f}MB")
        print("="*60)
    
    def save_detailed_report(self, metrics: PerformanceMetrics, filepath: str):
        """保存详细报告"""
        import json
        
        report = {
            "model_info": {
                "name": metrics.model_name,
                "device": metrics.device_info,
                "batch_size": metrics.batch_size
            },
            "timing": {
                "total_time": metrics.total_time,
                "avg_latency": metrics.avg_latency,
                "p50_latency": metrics.p50_latency,
                "p95_latency": metrics.p95_latency,
                "p99_latency": metrics.p99_latency,
                "time_to_first_token": metrics.time_to_first_token,
                "time_per_output_token": metrics.time_per_output_token,
                "latencies": metrics.latencies,
                "first_token_times": metrics.first_token_times
            },
            "throughput": {
                "tokens_per_second": metrics.tokens_per_second,
                "requests_per_second": metrics.requests_per_second,
                "token_counts": metrics.token_counts
            },
            "memory": {
                "gpu_memory_peak": metrics.gpu_memory_peak,
                "gpu_memory_used": metrics.gpu_memory_used,
                "cpu_memory_used": metrics.cpu_memory_used
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"详细报告已保存到: {filepath}")
    
    def save_markdown_report(self, metrics: PerformanceMetrics, config_dict: dict, 
                           filepath: str, method_name: str = "推理"):
        """保存Markdown格式报告"""
        import json
        from datetime import datetime
        
        report_md = f"""# {method_name}实验报告

## 实验配置
**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验名称**: {config_dict.get('experiment_name', 'default')}  
**优化技术**: {config_dict.get('technique', 'Unknown')}  
**批次级别**: {config_dict.get('batch_level', 'medium')}  

### 模型配置
- **模型**: {config_dict.get('model', 'Unknown')}
- **设备**: {config_dict.get('device', 'Unknown')}
- **批次大小**: {config_dict.get('batch_size', 'Unknown')}

### 推理参数
- **最大tokens**: {config_dict.get('max_tokens', 'Unknown')}
- **温度**: {config_dict.get('temperature', 'Unknown')}
- **Top-p**: {config_dict.get('top_p', 'Unknown')}

### 测试参数
- **提示数量**: {config_dict.get('num_prompts', 'Unknown')}
- **预热迭代**: {config_dict.get('warmup_iterations', 'Unknown')}
- **测试迭代**: {config_dict.get('test_iterations', 'Unknown')}

"""
        
        if config_dict.get('quantization_config'):
            report_md += f"""### 量化配置
```yaml
{json.dumps(config_dict['quantization_config'], indent=2)}
```

"""
        
        report_md += f"""## 性能指标结果

### 延迟指标
- **总时间**: {metrics.total_time:.3f}s
- **平均延迟**: {metrics.avg_latency*1000:.2f}ms
- **P50延迟**: {metrics.p50_latency*1000:.2f}ms
- **P95延迟**: {metrics.p95_latency*1000:.2f}ms
- **P99延迟**: {metrics.p99_latency*1000:.2f}ms
- **首token时间**: {metrics.time_to_first_token*1000:.2f}ms
- **每输出token时间**: {metrics.time_per_output_token*1000:.2f}ms

### 吞吐量指标
- **Tokens/秒**: {metrics.tokens_per_second:.2f}
- **请求/秒**: {metrics.requests_per_second:.2f}

### 内存使用
- **GPU内存峰值**: {metrics.gpu_memory_peak:.2f}MB
- **GPU内存当前**: {metrics.gpu_memory_used:.2f}MB
- **CPU内存**: {metrics.cpu_memory_used:.2f}MB

### 设备信息
- **设备**: {metrics.device_info}

## 原始数据

### 延迟数据 (ms)
```
{[round(lat*1000, 2) for lat in metrics.latencies[:10]]}{'...' if len(metrics.latencies) > 10 else ''}
```

### Token数量
```
{metrics.token_counts[:10]}{'...' if len(metrics.token_counts) > 10 else ''}
```

### 首token时间 (ms)
```
{[round(ft*1000, 2) for ft in metrics.first_token_times[:10]]}{'...' if len(metrics.first_token_times) > 10 else ''}
```

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_md)
            print(f"Markdown报告已保存到: {filepath}")
        except Exception as e:
            print(f"保存Markdown报告失败: {e}")


def compare_metrics(metrics1: PerformanceMetrics, metrics2: PerformanceMetrics, 
                   name1: str = "方法1", name2: str = "方法2"):
    """比较两个性能指标"""
    print("\n" + "="*80)
    print("性能对比")
    print("="*80)
    
    print(f"{'指标':<20} {'方法1':<15} {'方法2':<15} {'提升':<15}")
    print("-" * 80)
    
    # 延迟对比
    if metrics1.avg_latency > 0:
        latency_improvement = (metrics1.avg_latency - metrics2.avg_latency) / metrics1.avg_latency * 100
    else:
        latency_improvement = 0
    print(f"{'平均延迟(ms)':<20} {metrics1.avg_latency*1000:<15.2f} {metrics2.avg_latency*1000:<15.2f} {latency_improvement:<15.2f}%")
    
    # 吞吐量对比
    if metrics1.tokens_per_second > 0:
        throughput_improvement = (metrics2.tokens_per_second - metrics1.tokens_per_second) / metrics1.tokens_per_second * 100
    else:
        throughput_improvement = 0
    print(f"{'吞吐量(tokens/s)':<20} {metrics1.tokens_per_second:<15.2f} {metrics2.tokens_per_second:<15.2f} {throughput_improvement:<15.2f}%")
    
    # 内存对比
    if metrics1.gpu_memory_peak > 0:
        memory_improvement = (metrics1.gpu_memory_peak - metrics2.gpu_memory_peak) / metrics1.gpu_memory_peak * 100
        print(f"{'GPU内存峰值(MB)':<20} {metrics1.gpu_memory_peak:<15.2f} {metrics2.gpu_memory_peak:<15.2f} {memory_improvement:<15.2f}%")
    else:
        print(f"{'GPU内存峰值(MB)':<20} {metrics1.gpu_memory_peak:<15.2f} {metrics2.gpu_memory_peak:<15.2f} {'N/A':<15}")
    
    print("="*80)
    
    # 总结
    if throughput_improvement > 0:
        print(f"{name2} 在吞吐量方面比 {name1} 提升了 {throughput_improvement:.2f}%")
    elif throughput_improvement < 0:
        print(f"{name2} 在吞吐量方面比 {name1} 降低了 {abs(throughput_improvement):.2f}%")
    
    if latency_improvement > 0:
        print(f"{name2} 在延迟方面比 {name1} 改善了 {latency_improvement:.2f}%")
    elif latency_improvement < 0:
        print(f"{name2} 在延迟方面比 {name1} 增加了 {abs(latency_improvement):.2f}%")
