import asyncio
from retrieval.generator import answer_question
async def main():

    result = await answer_question(
        "What should patients with severe liver disease do before using paracetamol?",
        domain_filter="medical",
    )

    print("Question:", result.question)
    print("\nAnswer:\n", result.answer)
    print("\nCitations:")
    for c in result.citations:
        print("-", c.doc_title, "page", c.page_num, "score", c.relevance_score)

    print("\nModel:", result.model)
    print("Latency:", result.latency_ms, "ms")


asyncio.run(main())