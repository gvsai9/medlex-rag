import asyncio
from retrieval.generator import answer_question


async def main():
    result = await answer_question(
        "What are the warnings for ibuprofen?",
        domain_filter="fda",
    )

    print("Question:", result.question)
    print("\nAnswer:\n", result.answer)

    print("\nCitations:")
    for c in result.citations:
        print("-", c.doc_title, "page", c.page_num, "score", c.relevance_score)

    print("\nModel:", result.model)
    print("Latency:", result.latency_ms, "ms")


if __name__ == "__main__":
    asyncio.run(main())