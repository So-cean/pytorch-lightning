# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import shutil
import subprocess
from typing import Any, Optional, Union

import torch
import torch_npu
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.accelerators.npu import _check_cuda_matmul_precision, _clear_npu_memory, num_npu_devices
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.device_parser import _parse_npu_ids
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class NPUAccelerator(Accelerator):
    """Accelerator for Ascend NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        if device.type != "npu":
            raise MisconfigurationException(f"Device should be NPU, got {device} instead")
        _check_cuda_matmul_precision(device)
        #torch.cuda.set_device(device)
        torch.npu.set_device(device)

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        self.set_ascend_flags(trainer.local_rank)
        _clear_npu_memory()

    @staticmethod
    def set_ascend_flags(local_rank: int) -> None:
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_npu_ids = ",".join(str(x) for x in range(num_npu_devices()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_npu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - NPU_VISIBLE_DEVICES: [{devices}]")

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If nvidia-smi installation not found

        """
        return torch.npu.memory_stats(device)

    @override
    def teardown(self) -> None:
        _clear_npu_memory()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic."""
        return _parse_npu_ids(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("npu", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_npu_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return num_npu_devices() > 0

    @staticmethod
    @override
    def name() -> str:
        return "npu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
    
    @override
    def get_distribute_name(self) -> str:
        return "hccl"

    @override
    def get_stream_context(self, device_id: list[int]) -> Any:
        from contextlib import nullcontext
        return torch.npu.stream(torch.npu.Stream()) if device_id is not None else nullcontext()


def get_nvidia_gpu_stats(device: _DEVICE) -> dict[str, float]:  # pragma: no-cover
    """Get NPU stats including memory, power, and temperature from npu-smi.

    Returns
    -------
    dict[str, float]
        与旧 GPU 接口保持一致，key 仍叫 'utilization.gpu (%)' 等，
        方便上层代码不用改字段名。
    """
    npu_smi_path = shutil.which("npu-smi")
    if npu_smi_path is None:
        raise FileNotFoundError("npu-smi: command not found")

    # 拿到设备索引
    index = torch._utils._get_device_index(device)

    # 只抓当前卡的一行精简信息
    result = subprocess.run(
        [npu_smi_path, "info", "-t", "raw", "-i", str(index)],
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    # 原始输出示例（一行）：
    # 0  910B3  OK  98.9  36  0  / 0  0000:C1:00.0  0  0  / 0  5117  / 65536
    raw = result.stdout.strip().split()
    if len(raw) < 12:          # 防御
        return {}

    # 按列取数
    power_w = float(raw[3])          # Power(W)
    temp_c  = float(raw[4])          # Temp(C)
    aicore  = float(raw[11])         # AICore(%)
    mem_used_mb = float(raw[12])     # Memory-Usage(MB) 已用
    mem_total_mb = float(raw[14])    # Memory-Usage(MB) 总量
    mem_util = (mem_used_mb / mem_total_mb * 100) if mem_total_mb else 0.0

    # 与旧 GPU 字段保持同名，上层代码无需改动
    return {
        "utilization.gpu (%)": aicore,            # 对应 GPU util
        "memory.used (MB)": mem_used_mb,
        "memory.free (MB)": mem_total_mb - mem_used_mb,
        "utilization.memory (%)": mem_util,
        "fan.speed (%)": 0.0,                     # NPU 无风扇，填 0
        "temperature.gpu (°C)": temp_c,
        "temperature.memory (°C)": 0.0,           # 暂无 memory 传感器
    }
