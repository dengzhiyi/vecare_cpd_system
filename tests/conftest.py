"""让仓库根目录在任意目录下运行 pytest 时都可被导入。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
