"""
推理模块基类 - InferenceModule

该模块定义了推理模块的基础接口，所有具体的推理模块都应该继承这个基类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from performance_monitor import PerformanceMonitor, PerformanceMetrics
from config import Config


class InferenceModule(ABC):
    """推理模块基类"""
    
    def __init__(self, config: Config):
        """
        初始化推理模块
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor()
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
        # 重置监测器
        self.performance_monitor.reset()
        
        # 预热阶段
        print(f"预热阶段...")
        warmup_prompts = prompts[:min(len(prompts), self.config.warmup_iterations)]
        for _ in range(self.config.warmup_iterations):
            _ = inference_func(warmup_prompts)
        
        # 正式测试
        print(f"正式测试阶段...")
        all_outputs = []
        self.performance_monitor.start_monitoring()
        
        for i in range(self.config.test_iterations):
            with self.performance_monitor.measure_batch():
                outputs = inference_func(prompts)
                all_outputs.extend(outputs)
                
                # 记录token数量
                total_tokens = sum(len(output.split()) for output in outputs)
                self.performance_monitor.add_token_count(total_tokens)
        
        self.performance_monitor.stop_monitoring()
        
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
        打印性能监测结果
        
        Args:
            metrics: 性能指标
            method_name: 方法名称
        """
        print(f"\n{method_name} 性能结果:")
        self.performance_monitor.print_summary(metrics)
        
        # 保存详细报告
        if self.config.save_results:
            filename = f"{method_name.replace(' ', '_').lower()}_report.json"
            filepath = f"{self.config.output_dir}/{filename}"
            self.performance_monitor.save_detailed_report(metrics, filepath)
    
    def compare_results(self, basic_metrics: PerformanceMetrics, 
                       optimized_metrics: PerformanceMetrics):
        """
        比较基础推理和优化推理的结果
        
        Args:
            basic_metrics: 基础推理性能指标
            optimized_metrics: 优化推理性能指标
        """
        from performance_monitor import compare_metrics
        compare_metrics(basic_metrics, optimized_metrics, "基础推理", "优化推理")
    
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
