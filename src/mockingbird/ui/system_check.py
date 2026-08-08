"""System capability checks — warn user about missing CUDA, no LLM, no KB modules, etc."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from mockingbird.config import Config

log = logging.getLogger(__name__)


@dataclass
class SystemWarning:
    level: str  # "error", "warning", "info"
    title: str
    message: str


def run_system_checks(config: Config) -> list[SystemWarning]:
    """Check system capabilities and return a list of warnings.

    Called at startup (after App init) and surfaced to the user via
    QMessageBox or status bar.
    """
    warnings: list[SystemWarning] = []

    # 1. LLM not configured
    if not config.llm.base_url or not config.llm.api_key:
        warnings.append(SystemWarning(
            level="error",
            title="LLM не настроен",
            message="Без LLM-подключения ответы, контекст-анализ и personal-режим не работают. "
            "Настройте в «Настройки → LLM».",
        ))

    # 2. No CUDA (GPU) — STT will be slower
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if not cuda_available and config.stt.backend == "gigaam":
            warnings.append(SystemWarning(
                level="warning",
                title="GPU (CUDA) недоступен",
                message="GigaAM на CPU работает значительно медленнее. "
                "Рассмотрите whisper с compute_type=int8 для скорости или включите CUDA.",
            ))
    except Exception:
        # torch not installed yet, or CUDA driver mismatch / load error — the
        # ``ImportError`` subclass is too narrow (torch.cuda.is_available can
        # raise RuntimeError/OSError on a driver mismatch); fall back to a
        # generic warning so the app still boots on CPU.
        logging.getLogger(__name__).debug("torch/CUDA probe failed", exc_info=True)

    # 3. No active KB modules
    try:
        from mockingbird.kb.module_manager import ModuleManager
        from mockingbird.kb.loader import load_topics

        topics = load_topics(config.interview.kb_path)
        if not topics:
            warnings.append(SystemWarning(
                level="warning",
                title="База знаний пуста",
                message="Нет активных модулей базы знаний. Импортируйте модуль в «Настройки → Модули».",
            ))
    except Exception:
        pass

    # 4. No resume loaded (personal mode)
    try:
        from mockingbird.kb.resume_loader import ResumeLoader

        if not ResumeLoader.is_loaded():
            warnings.append(SystemWarning(
                level="info",
                title="Резюме не загружено",
                message="Personal-вопросы («что ты делал?») будут отвечать в constructive-режиме "
                "(без конкретных фактов из резюме). Загрузите PDF в «Настройки → Резюме».",
            ))
    except Exception:
        pass

    # 5. Capture protection unavailable
    try:
        from mockingbird.ui import capture_guard

        if (
            capture_guard.is_supported()
            and not capture_guard.is_capture_protection_available()
        ):
            warnings.append(SystemWarning(
                level="info",
                title="Скрытие от захвата недоступно",
                message=f"Требуется Windows 10 build 19041+. "
                f"У вас build {capture_guard.windows_build() or '?'}.",
            ))
    except Exception:
        pass

    return warnings
