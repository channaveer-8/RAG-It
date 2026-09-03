import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


from rag.llm.ollama_manager import (
    OllamaManager,
)


def main():

    print("=" * 70)
    print("OLLAMA MANAGER TEST")
    print("=" * 70)

    manager = OllamaManager(
        model_name="qwen3.5:4b-q4_K_M",
    )

    manager.ensure_ready()

    print()
    print("=" * 70)
    print("OLLAMA MANAGER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()