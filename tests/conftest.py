import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Araç testleri LLM çağırmaz; ayarların yüklenebilmesi için sahte değerler yeter
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:1/v1")
os.environ.setdefault("LLM_API_KEY", "test")
