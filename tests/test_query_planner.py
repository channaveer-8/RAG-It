import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SRC_DIR = (
    PROJECT_ROOT / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)


from rag.retrieval.query_planner import (
    QueryPlanner,
)


def main():

    print("=" * 70)
    print("QUERY PLANNER TEST")
    print("=" * 70)

    question = (
        "I want to connect my STM32F4 board "
        "to OpenOCD and debug it using breakpoints. "
        "Give me step-by-step commands and explanation."
    )

    print("\nUSER QUESTION:")
    print(question)

    planner = QueryPlanner()

    try:

        queries = planner.plan(
            question
        )

    except Exception as exc:

        print(
            "\n[ERROR]"
        )

        print(
            exc
        )

        return

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL QUERIES"
    )

    print(
        "=" * 70
    )

    for index, query in enumerate(
        queries,
        start=1,
    ):

        print(
            f"{index}. {query}"
        )

    print(
        "\nTotal queries:",
        len(queries),
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()