"""
配置模块 - Config

该模块用来管理项目中的各项配置，包括模型路径、设备配置等。

注意：本项目已配置为统一使用 DeepSeek-R1-Distill-Qwen-1.5B 模型，
所有预设配置和默认配置都已更新为使用此模型，确保整个推理工作流的一致性。
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """项目配置类"""
    
    # 模型配置
    model: str = "/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B"  # 默认使用小模型进行测试
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
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.device is None:
            # 自动检测可用的CUDA设备
            import torch
            if torch.cuda.is_available():
                self.device = list(range(torch.cuda.device_count()))
            else:
                self.device = []
        
        # 创建输出目录
        if self.save_results and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
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


# 预定义的配置
class ConfigPresets:
    """预定义配置"""
    
    @staticmethod
    def small_model_config() -> Config:
        """小模型配置，用于快速测试"""
        return Config(
            model="/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B",
            max_tokens=50,
            num_prompts=5,
            test_iterations=3
        )
    
    @staticmethod
    def large_model_config() -> Config:
        """大模型配置，用于性能测试"""
        return Config(
            model="/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B",
            max_tokens=200,
            num_prompts=20,
            test_iterations=10
        )


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
