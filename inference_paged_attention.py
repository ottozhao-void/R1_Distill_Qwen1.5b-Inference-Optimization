"""
PagedAttention推理模块 - InferenceOnPagedAttention

使用vLLM库实现基于PagedAttention的模型推理，对比组是基于transformers库的相同模型的推理。
这构成了针对PagedAttention优化技术的对比实验。
"""

from typing import List, Optional, Any, Dict
import gc
import warnings
warnings.filterwarnings("ignore")

from config import Config
from inference_module import InferenceModule


class InferenceOnPagedAttention(InferenceModule):
    """基于PagedAttention的推理模块"""
    
    def __init__(self, config: Config):
        """
        初始化PagedAttention推理模块
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        
        # 延迟初始化模型
        self.transformers_model = None
        self.transformers_tokenizer = None
        self.vllm_model = None
        
        print(f"初始化PagedAttention推理模块")
        print(f"模型: {config.model}")
        print(f"设备: {config.get_device_str()}")
        print(f"是否多GPU: {config.is_multi_gpu()}")
    
    def _initialize_transformers_model(self):
        """初始化transformers模型"""
        if self.transformers_model is not None:
            return
        
        print("正在加载transformers模型...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # 加载tokenizer
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            if self.transformers_tokenizer.pad_token is None:
                self.transformers_tokenizer.pad_token = self.transformers_tokenizer.eos_token
            
            # 加载模型
            device_map = "auto" if self.config.is_multi_gpu() else self.config.get_device_str()
            
            self.transformers_model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                device_map=device_map,
                torch_dtype=torch.float16 if "cuda" in self.config.get_device_str() else torch.float32,
                trust_remote_code=True
            )
            
            print(f"transformers模型加载完成，使用设备: {device_map}")
            
        except Exception as e:
            print(f"加载transformers模型失败: {e}")
            raise
    
    def _initialize_vllm_model(self):
        """初始化vLLM模型"""
        if self.vllm_model is not None:
            return
        
        print("正在加载vLLM模型...")
        try:
            from vllm import LLM
            
            # vLLM初始化参数
            vllm_kwargs = {
                "model": self.config.model,
                "trust_remote_code": True,
                "max_model_len": 2048,  # 限制序列长度以节省内存
                "enforce_eager": True,  # 禁用torch编译以避免ldconfig问题
            }
            
            # 设置GPU相关参数
            if self.config.device and len(self.config.device) > 0:
                vllm_kwargs["tensor_parallel_size"] = self.config.get_tensor_parallel_size()
            
            # 如果只有一个GPU或CPU，设置相应参数
            if self.config.device and len(self.config.device) == 1:
                vllm_kwargs["gpu_memory_utilization"] = 0.5  # Reduced to avoid conflicts
            
            self.vllm_model = LLM(**vllm_kwargs)
            
            print(f"vLLM模型加载完成，张量并行度: {self.config.get_tensor_parallel_size()}")
            
        except Exception as e:
            print(f"加载vLLM模型失败: {e}")
            raise
    
    def basic_inference(self, prompts: List[str]) -> List[str]:
        """
        基于transformers库的基础推理（对照组）
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            生成的文本列表
        """
        self._initialize_transformers_model()
        
        try:
            import torch
            
            outputs = []
            
            for prompt in prompts:
                # 编码输入
                inputs = self.transformers_tokenizer(
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
                    generated_ids = self.transformers_model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.transformers_tokenizer.eos_token_id,
                        eos_token_id=self.transformers_tokenizer.eos_token_id,
                    )
                
                # 解码输出
                generated_text = self.transformers_tokenizer.decode(
                    generated_ids[0], 
                    skip_special_tokens=True
                )
                
                # 移除原始prompt，只保留生成的部分
                generated_only = generated_text[len(prompt):].strip()
                outputs.append(generated_only)
            
            return outputs
            
        except Exception as e:
            print(f"transformers推理失败: {e}")
            return [f"Error: {e}"] * len(prompts)
    
    def optimized_inference(self, prompts: List[str]) -> List[str]:
        """
        基于vLLM库的优化推理（实验组）
        使用PagedAttention优化内存管理
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            生成的文本列表
        """
        self._initialize_vllm_model()
        
        try:
            from vllm import SamplingParams
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            
            # 使用vLLM进行批量推理
            outputs = self.vllm_model.generate(prompts, sampling_params)
            
            # 提取生成的文本
            generated_texts = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            print(f"vLLM推理失败: {e}")
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
                if self.transformers_model is not None:
                    del self.transformers_model
                    self.transformers_model = None
                if self.transformers_tokenizer is not None:
                    del self.transformers_tokenizer
                    self.transformers_tokenizer = None
                
                import gc
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
        
        # 清理transformers模型
        if self.transformers_model is not None:
            del self.transformers_model
            self.transformers_model = None
        
        if self.transformers_tokenizer is not None:
            del self.transformers_tokenizer
            self.transformers_tokenizer = None
        
        # 清理vLLM模型
        if self.vllm_model is not None:
            del self.vllm_model
            self.vllm_model = None
        
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
        print("PagedAttention vs Traditional Key-Value Cache 性能对比测试")
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
        print(f"  ✓ 基础推理 (transformers): 平均延迟 {basic_metrics.avg_latency*1000:.2f}ms")
        print(f"  ✓ 优化推理 (vLLM+PagedAttention): 平均延迟 {optimized_metrics.avg_latency*1000:.2f}ms")
        
        # 计算改进
        latency_improvement = (basic_metrics.avg_latency - optimized_metrics.avg_latency) / basic_metrics.avg_latency * 100
        throughput_improvement = (optimized_metrics.tokens_per_second - basic_metrics.tokens_per_second) / basic_metrics.tokens_per_second * 100
        memory_improvement = (basic_metrics.gpu_memory_peak - optimized_metrics.gpu_memory_peak) / basic_metrics.gpu_memory_peak * 100
        
        print(f"\nPagedAttention优化效果:")
        if latency_improvement > 0:
            print(f"  🚀 延迟降低: {latency_improvement:.1f}%")
        if throughput_improvement > 0:
            print(f"  📈 吞吐量提升: {throughput_improvement:.1f}%")
        if memory_improvement > 0:
            print(f"  💾 内存节省: {memory_improvement:.1f}%")
        
        print(f"\n结论:")
        if throughput_improvement > 10:
            print(f"  ✅ PagedAttention显著提升了推理性能")
        elif throughput_improvement > 0:
            print(f"  ✅ PagedAttention适度提升了推理性能")
        else:
            print(f"  ⚠️  在当前测试场景下，优化效果不明显")
        
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
            
            # 临时修改配置
            original_iterations = self.config.test_iterations
            self.config.test_iterations = 2  # 减少迭代次数以节省时间
            
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
