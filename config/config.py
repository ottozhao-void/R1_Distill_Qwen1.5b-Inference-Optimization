"""
配置模块 - Config

该模块用来管理项目中的各项配置，包括模型路径、设备配置等。
支持从YAML文件加载实验配置，每个优化技术有独立的配置文件。

注意：本项目已配置为统一使用 DeepSeek-R1-Distill-Qwen-1.5B 模型，
所有配置都已更新为使用此模型，确保整个推理工作流的一致性。
"""

import os
import yaml
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Config:
    """项目配置类"""
    
    # 实验配置
    experiment_name: str = "default"
    technique: str = "PagedAttention"  # PagedAttention, Quantization
    batch_level: str = "medium"  # small, medium, large
    
    # 模型配置
    model: str = "/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B"
    device: Optional[List[int]] = field(default_factory=lambda: [4])  # CUDA设备ID列表，None表示自动检测
    
    # 推理配置
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95
    batch_size: int = 4
    
    # 性能测试配置
    num_prompts: int = 10
    warmup_iterations: int = 3
    test_iterations: int = 5
    
    # 输出配置
    output_dir: str = "results"
    save_results: bool = True
    
    # 量化相关配置
    quantization_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.device is None:
            # 自动检测可用的CUDA设备
            import torch
            if torch.cuda.is_available():
                self.device = list(range(torch.cuda.device_count()))
            else:
                self.device = []
        
        # 根据batch_level调整batch_size
        if self.batch_level == "small":
            self.batch_size = min(self.batch_size, 2)
            self.num_prompts = min(self.num_prompts, 5)
        elif self.batch_level == "large":
            self.batch_size = max(self.batch_size, 8)
            self.num_prompts = max(self.num_prompts, 20)
        
        # 创建输出目录（根据技术类型、批次级别和时间戳）
        if self.save_results:
            technique_dir = os.path.join("results", self.technique)
            if not os.path.exists(technique_dir):
                os.makedirs(technique_dir)
            
            # 只有在output_dir还是默认值时才更新
            if self.output_dir == "results":
                # 生成时间戳字符串
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 创建带批次级别和时间戳的目录
                batch_timestamp_dir = os.path.join(technique_dir, f"{self.batch_level}_{timestamp}")
                if not os.path.exists(batch_timestamp_dir):
                    os.makedirs(batch_timestamp_dir)
                
                self.output_dir = batch_timestamp_dir
    
    def get_tensor_parallel_size(self) -> int:
        """获取张量并行大小"""
        return len(self.device) if self.device else 1
    
    def get_device_str(self) -> str:
        """获取设备字符串"""
        if not self.device:
            return "cpu"
        elif len(self.device) == 1:
            return f"cuda:{self.device[0]}"
        else:
            return f"cuda:{self.device[0]}"  # 主设备
    
    def is_multi_gpu(self) -> bool:
        """是否使用多GPU"""
        return len(self.device) > 1 if self.device else False


def load_config_from_yaml(config_path: str) -> Config:
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 创建Config对象
        config = Config()
        
        # 更新配置字段
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 手动调用__post_init__进行后处理
        config.__post_init__()
        
        return config
        
    except Exception as e:
        print(f"从YAML文件加载配置失败: {e}")
        print("使用默认配置")
        return Config()


def save_config_to_yaml(config: Config, config_path: str):
    """保存配置到YAML文件"""
    try:
        # 创建目录
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 转换为字典
        config_dict = {
            'experiment_name': config.experiment_name,
            'technique': config.technique,
            'batch_level': config.batch_level,
            'model': config.model,
            'device': config.device,
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'batch_size': config.batch_size,
            'num_prompts': config.num_prompts,
            'warmup_iterations': config.warmup_iterations,
            'test_iterations': config.test_iterations,
            'output_dir': config.output_dir,
            'save_results': config.save_results,
            'quantization_config': config.quantization_config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置已保存到: {config_path}")
        
    except Exception as e:
        print(f"保存配置到YAML文件失败: {e}")


def get_config_for_technique(technique: str, batch_level: str = "medium", 
                           experiment_name: str = "default") -> Config:
    """根据优化技术获取配置"""
    config_dir = "config"
    config_filename = f"{technique.lower()}_{batch_level}_{experiment_name}.yaml"
    config_path = os.path.join(config_dir, technique, config_filename)
    
    if os.path.exists(config_path):
        return load_config_from_yaml(config_path)
    else:
        # 创建默认配置
        config = Config()
        config.technique = technique
        config.batch_level = batch_level
        config.experiment_name = experiment_name
        
        # 保存默认配置
        save_config_to_yaml(config, config_path)
        return config


def get_available_devices() -> Dict[str, Any]:
    """获取可用设备信息"""
    device_info = {"has_cuda": False, "cuda_devices": [], "cpu_available": True}
    
    try:
        import torch
        if torch.cuda.is_available():
            device_info["has_cuda"] = True
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                device_info["cuda_devices"].append({
                    "id": i,
                    "name": device_name,
                    "memory_gb": round(memory_total, 1)
                })
    except ImportError:
        pass
    
    return device_info


def print_available_devices():
    """打印可用设备信息"""
    device_info = get_available_devices()
    
    print("🖥️  可用设备信息:")
    print("="*50)
    
    if device_info["has_cuda"]:
        print("✅ CUDA 设备:")
        for device in device_info["cuda_devices"]:
            print(f"   GPU {device['id']}: {device['name']} ({device['memory_gb']}GB)")
        print(f"\n💡 使用示例:")
        print(f"   --device 0          # 使用 GPU 0")
        if len(device_info["cuda_devices"]) > 1:
            print(f"   --device 0,1        # 使用 GPU 0 和 1")
    else:
        print("❌ 未检测到 CUDA 设备")
    
    if device_info["cpu_available"]:
        print(f"✅ CPU 可用")
        print(f"   --device cpu        # 使用 CPU")
    
    print("="*50)


def load_config_from_env() -> Config:
    """从环境变量加载配置"""
    config = Config()
    
    # 从环境变量读取配置
    model_name = os.getenv("MODEL_NAME")
    if model_name:
        config.model = model_name
    
    cuda_devices = os.getenv("CUDA_DEVICES")
    if cuda_devices:
        device_ids = [int(x) for x in cuda_devices.split(",")]
        config.device = device_ids
    
    max_tokens = os.getenv("MAX_TOKENS")
    if max_tokens:
        config.max_tokens = int(max_tokens)
    
    temperature = os.getenv("TEMPERATURE")
    if temperature:
        config.temperature = float(temperature)
    
    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        config.output_dir = output_dir
    
    return config
