"""
配置模块 - Config

该模块用来管理项目中的各项配置，包括模型路径、设备配置等。
"""

import os
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """项目配置类"""
    
    # 模型配置
    model: str = "facebook/opt-125m"  # 默认使用小模型进行测试
    device: List[int] = None  # CUDA设备ID列表，None表示自动检测
    
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
            model="facebook/opt-125m",
            max_tokens=50,
            num_prompts=5,
            test_iterations=3
        )
    
    @staticmethod
    def large_model_config() -> Config:
        """大模型配置，用于性能测试"""
        return Config(
            model="facebook/opt-1.3b",
            max_tokens=200,
            num_prompts=20,
            test_iterations=10
        )
    
    @staticmethod
    def llama_config() -> Config:
        """Llama模型配置"""
        return Config(
            model="meta-llama/Llama-2-7b-chat-hf",
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            num_prompts=15,
            test_iterations=10
        )


def load_config_from_env() -> Config:
    """从环境变量加载配置"""
    config = Config()
    
    # 从环境变量读取配置
    if os.getenv("MODEL_NAME"):
        config.model = os.getenv("MODEL_NAME")
    
    if os.getenv("CUDA_DEVICES"):
        device_ids = [int(x) for x in os.getenv("CUDA_DEVICES").split(",")]
        config.device = device_ids
    
    if os.getenv("MAX_TOKENS"):
        config.max_tokens = int(os.getenv("MAX_TOKENS"))
    
    if os.getenv("TEMPERATURE"):
        config.temperature = float(os.getenv("TEMPERATURE"))
    
    if os.getenv("OUTPUT_DIR"):
        config.output_dir = os.getenv("OUTPUT_DIR")
    
    return config
