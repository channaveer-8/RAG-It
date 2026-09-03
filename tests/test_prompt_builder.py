import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from rag.prompt import PromptBuilder


def main():

    print("=" * 70)
    print("PROMPT BUILDER TEST")
    print("=" * 70)

    rag_prompt = (
        "You are a technical document assistant. "
        "Prioritize information from the retrieved "
        "documents. Include units when available. "
        "Do not infer values that are not present."
    )

    builder = PromptBuilder(
        rag_prompt=rag_prompt,
    )

    context = """
[Context 1]
Document: vehicle_paper.pdf
Page: 2
Type: table

Battery nominal voltage: 525.4 V
Battery capacity: 296 Ah
"""

    query = (
        "What is the nominal voltage "
        "of the battery?"
    )

    prompt = builder.build(
        query=query,
        context=context,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATED PROMPT"
    )

    print(
        "=" * 70
    )

    print()
    print(prompt)

    print(
        "\n" + "=" * 70
    )

    print(
        "PROMPT BUILDER TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()