"""
量化推理模块 - InferenceOnQuantization

使用transformers库实现模型量化的对比实验，对比组是标准精度推理，实验组是量化推理。
这构成了针对量化优化技术的对比实验。
"""

from typing import List, Optional, Any, Dict
import gc
import warnings
warnings.filterwarnings("ignore")

from config.config import Config
from inference.inference_module import InferenceModule


class InferenceOnQuantization(InferenceModule):
    """基于量化的推理模块"""
    
    def __init__(self, config: Config):
        """
        初始化量化推理模块
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        
        # 延迟初始化模型
        self.standard_model = None
        self.standard_tokenizer = None
        self.quantized_model = None
        self.quantized_tokenizer = None
        
        print(f"初始化量化推理模块")
        print(f"模型: {config.model}")
        print(f"设备: {config.get_device_str()}")
        print(f"量化配置: {config.quantization_config}")
    
    def _initialize_standard_model(self):
        """初始化标准精度模型"""
        if self.standard_model is not None:
            return
        
        print("正在加载标准精度模型...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # 加载tokenizer
            self.standard_tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            if self.standard_tokenizer.pad_token is None:
                self.standard_tokenizer.pad_token = self.standard_tokenizer.eos_token
            
            # 加载标准精度模型
            device_map = "auto" if self.config.is_multi_gpu() else self.config.get_device_str()
            
            self.standard_model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                device_map=device_map,
                torch_dtype=torch.float16 if "cuda" in self.config.get_device_str() else torch.float32,
                trust_remote_code=True
            )
            
            print(f"标准精度模型加载完成，使用设备: {device_map}")
            
        except Exception as e:
            print(f"加载标准精度模型失败: {e}")
            raise
    
    def _initialize_quantized_model(self):
        """初始化量化模型"""
        if self.quantized_model is not None:
            return
        
        print("正在加载量化模型...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            # 加载tokenizer
            self.quantized_tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            if self.quantized_tokenizer.pad_token is None:
                self.quantized_tokenizer.pad_token = self.quantized_tokenizer.eos_token
            
            # 配置量化参数
            quantization_config = None
            if self.config.quantization_config:
                quant_config = self.config.quantization_config
                
                if quant_config.get('load_in_4bit', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')),
                        bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
                        bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4')
                    )
                elif quant_config.get('load_in_8bit', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
            
            # 加载量化模型
            device_map = "auto" if self.config.is_multi_gpu() else self.config.get_device_str()
            
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            else:
                # 如果没有量化配置，使用标准float16
                model_kwargs["torch_dtype"] = torch.float16 if "cuda" in self.config.get_device_str() else torch.float32
            
            self.quantized_model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                **model_kwargs
            )
            
            print(f"量化模型加载完成，使用设备: {device_map}")
            if quantization_config:
                print(f"量化配置: {quantization_config}")
            
        except Exception as e:
            print(f"加载量化模型失败: {e}")
            raise
    
    def basic_inference(self, prompts: List[str]) -> List[str]:
        """
        基于标准精度的基础推理（对照组）
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            生成的文本列表
        """
        self._initialize_standard_model()
        
        try:
            import torch
            
            outputs = []
            
            for prompt in prompts:
                # 记录首token生成时间
                with self.performance_monitor.measure_first_token():
                    # 编码输入
                    inputs = self.standard_tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=512
                    )
                    
                    # 移动到设备
                    device = self.config.get_device_str()
                    if device != "cpu":
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 生成文本
                    with torch.no_grad():
                        generated_ids = self.standard_model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=True,
                            pad_token_id=self.standard_tokenizer.eos_token_id,
                            eos_token_id=self.standard_tokenizer.eos_token_id,
                        )
                
                # 解码输出
                generated_text = self.standard_tokenizer.decode(
                    generated_ids[0], 
                    skip_special_tokens=True
                )
                
                # 移除原始prompt，只保留生成的部分
                generated_only = generated_text[len(prompt):].strip()
                outputs.append(generated_only)
            
            return outputs
            
        except Exception as e:
            print(f"标准精度推理失败: {e}")
            return [f"Error: {e}"] * len(prompts)
    
    def optimized_inference(self, prompts: List[str]) -> List[str]:
        """
        基于量化的优化推理（实验组）
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            生成的文本列表
        """
        self._initialize_quantized_model()
        
        try:
            import torch
            
            outputs = []
            
            for prompt in prompts:
                # 记录首token生成时间
                with self.performance_monitor.measure_first_token():
                    # 编码输入
                    inputs = self.quantized_tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=512
                    )
                    
                    # 移动到设备
                    device = self.config.get_device_str()
                    if device != "cpu":
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 生成文本
                    with torch.no_grad():
                        generated_ids = self.quantized_model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=True,
                            pad_token_id=self.quantized_tokenizer.eos_token_id,
                            eos_token_id=self.quantized_tokenizer.eos_token_id,
                        )
                
                # 解码输出
                generated_text = self.quantized_tokenizer.decode(
                    generated_ids[0], 
                    skip_special_tokens=True
                )
                
                # 移除原始prompt，只保留生成的部分
                generated_only = generated_text[len(prompt):].strip()
                outputs.append(generated_only)
            
            return outputs
            
        except Exception as e:
            print(f"量化推理失败: {e}")
            return [f"Error: {e}"] * len(prompts)
    
    def inference(self, prompts: List[str], method: str = "both") -> Dict[str, Any]:
        """
        重写推理方法以支持设备间的内存管理
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
            
            # 如果要运行两种方法，在切换到优化推理前清理基础推理模型
            if method == "both":
                print("清理基础推理模型以释放内存...")
                if self.standard_model is not None:
                    del self.standard_model
                    self.standard_model = None
                if self.standard_tokenizer is not None:
                    del self.standard_tokenizer
                    self.standard_tokenizer = None
                
                gc.collect()
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("CUDA缓存已清理")
                except ImportError:
                    pass
        
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
    
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        # 清理标准模型
        if self.standard_model is not None:
            del self.standard_model
            self.standard_model = None
        
        if self.standard_tokenizer is not None:
            del self.standard_tokenizer
            self.standard_tokenizer = None
        
        # 清理量化模型
        if self.quantized_model is not None:
            del self.quantized_model
            self.quantized_model = None
        
        if self.quantized_tokenizer is not None:
            del self.quantized_tokenizer
            self.quantized_tokenizer = None
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA缓存已清理")
        except ImportError:
            pass
        
        print("资源清理完成")
    
    def run_comparison_test(self, custom_prompts: Optional[List[str]] = None) -> dict:
        """
        运行完整的对比测试
        
        Args:
            custom_prompts: 自定义提示列表，如果为None则使用默认提示
            
        Returns:
            测试结果字典
        """
        print("\n" + "="*80)
        print("量化 vs 标准精度 性能对比测试")
        print("="*80)
        
        # 准备测试数据
        if custom_prompts is None:
            prompts = self.generate_test_prompts()
        else:
            prompts = custom_prompts
        
        print(f"测试配置:")
        print(f"  模型: {self.config.model}")
        print(f"  设备: {self.config.get_device_str()}")
        print(f"  提示数量: {len(prompts)}")
        print(f"  最大生成tokens: {self.config.max_tokens}")
        print(f"  测试迭代次数: {self.config.test_iterations}")
        print(f"  预热迭代次数: {self.config.warmup_iterations}")
        print(f"  量化配置: {self.config.quantization_config}")
        
        try:
            # 运行对比测试
            results = self.inference(prompts, method="both")
            
            # 生成总结报告
            self._generate_summary_report(results, prompts)
            
            return results
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            return {"error": str(e)}
        finally:
            # 确保资源被清理
            self.cleanup()
    
    def _generate_summary_report(self, results: dict, prompts: List[str]):
        """生成总结报告"""
        if "basic" not in results or "optimized" not in results:
            return
        
        basic_metrics = results["basic"]["metrics"]
        optimized_metrics = results["optimized"]["metrics"]
        
        print(f"\n" + "="*80)
        print("测试总结")
        print("="*80)
        
        print(f"测试概述:")
        print(f"  ✓ 基础推理 (标准精度): 平均延迟 {basic_metrics.avg_latency*1000:.2f}ms")
        print(f"  ✓ 优化推理 (量化): 平均延迟 {optimized_metrics.avg_latency*1000:.2f}ms")
        
        # 计算改进
        latency_improvement = (basic_metrics.avg_latency - optimized_metrics.avg_latency) / basic_metrics.avg_latency * 100
        throughput_improvement = (optimized_metrics.tokens_per_second - basic_metrics.tokens_per_second) / basic_metrics.tokens_per_second * 100
        memory_improvement = (basic_metrics.gpu_memory_peak - optimized_metrics.gpu_memory_peak) / basic_metrics.gpu_memory_peak * 100
        
        print(f"\n量化优化效果:")
        if latency_improvement > 0:
            print(f"  🚀 延迟降低: {latency_improvement:.1f}%")
        if throughput_improvement > 0:
            print(f"  📈 吞吐量提升: {throughput_improvement:.1f}%")
        if memory_improvement > 0:
            print(f"  💾 内存节省: {memory_improvement:.1f}%")
        
        print(f"\n结论:")
        if memory_improvement > 20:
            print(f"  ✅ 量化显著节省了内存使用")
        elif memory_improvement > 0:
            print(f"  ✅ 量化适度节省了内存使用")
        else:
            print(f"  ⚠️  在当前测试场景下，内存节省效果不明显")
        
        if throughput_improvement > 10:
            print(f"  ✅ 量化显著提升了推理性能")
        elif throughput_improvement > 0:
            print(f"  ✅ 量化适度提升了推理性能")
        else:
            print(f"  ⚠️  量化可能在速度上有轻微损失，但节省了内存")
        
        print("="*80)
    
    def benchmark_memory_efficiency(self, batch_sizes: Optional[List[int]] = None) -> dict:
        """
        测试不同批次大小下的内存效率
        
        Args:
            batch_sizes: 要测试的批次大小列表
            
        Returns:
            内存效率测试结果
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        
        print(f"\n内存效率基准测试")
        print(f"测试批次大小: {batch_sizes}")
        
        results = {}
        base_prompts = self.generate_test_prompts()
        
        for batch_size in batch_sizes:
            if batch_size > len(base_prompts):
                prompts = base_prompts * ((batch_size // len(base_prompts)) + 1)
                prompts = prompts[:batch_size]
            else:
                prompts = base_prompts[:batch_size]
            
            print(f"\n测试批次大小: {batch_size}")
            
            original_iterations = self.config.test_iterations
            self.config.test_iterations = 2 
            
            try:
                batch_results = self.inference(prompts, method="both")
                results[batch_size] = batch_results
                
            except Exception as e:
                print(f"批次大小 {batch_size} 测试失败: {e}")
                results[batch_size] = {"error": str(e)}
            finally:
                self.config.test_iterations = original_iterations
                self.cleanup()
        
        return results
