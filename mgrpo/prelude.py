"""公共导入：libriichi 路径、日志格式、torch 告警抑制"""
import sys
import logging
import warnings
from pathlib import Path

_MORTAL_DIR = Path(__file__).resolve().parent.parent / 'mortal'
if str(_MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(_MORTAL_DIR))  # libriichi.pyd 位于 mortal/ 下

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)

warnings.simplefilter('ignore')
import torch  # noqa: E402,F401  触发 numpy 告警后由下方恢复
import torch.utils.tensorboard  # noqa: E402,F401  distutils 弃用告警
warnings.simplefilter('default')
