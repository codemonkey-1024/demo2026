#!/usr/bin/env python
"""Train Queryflow with the public relative-path configuration."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QUERYFLOW_CONFIG_NAME", "default")

from queryflow.workflow.stage2_qa_finetune import main  # noqa: E402


if __name__ == "__main__":
    main()
