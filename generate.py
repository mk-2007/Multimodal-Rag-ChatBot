"""
generate.py
-----------
STEP 7 of the pipeline: generate a grounded answer using ONLY the retrieved
context, via Gemini's text generation. The system prompt explicitly instructs
the model to answer solely from the provided context (not from its own general
knowledge about the Transformer paper, which it likely has memorized) and to
say so if the context is insufficient -- this is what makes the pipeline a
genuine RAG rather than the LLM just answering from parametric memory.
"""

from google import genai

from config import require_api_key, GEMINI_GENERATION_MODEL
from retrieval import RetrievedChunk

SYSTEM_INSTRUCTION = """You are a research-paper question-answering assistant. \
You must answer the user's question USING ONLY the CONTEXT provided below, which was \
retrieved from a multimodal RAG system indexing the paper "Attention Is All You Need". \
The context may come from body text, a transcribed table, or a vision-generated \
description of a figure -- each labeled with its modality and source.

Rules:
- Ground every claim in the provided context. Do not use outside/general knowledge, \
even if you happen to know the paper well.
- If the context does not contain enough information to answer, say so explicitly \
rather than guessing.
- When the context includes a table or figure, feel free to cite it by name (e.g. \
"Table 2" or "Figure 1") since the user can see the underlying source.
- Be concise and directly answer the question first, then add supporting detail.
"""


def make_client() -> genai.Client:
    return genai.Client(api_key=require_api_key())


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"--- Context {i} | modality={c.modality} | source={c.source_ref} "
            f"| page={c.page} ---\n{c.content}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: list[RetrievedChunk], client: genai.Client = None) -> str:
    client = client or make_client()
    context_str = format_context(chunks)
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{context_str}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )
    response = client.models.generate_content(
        model=GEMINI_GENERATION_MODEL,
        contents=prompt,
    )
    return response.text.strip()


if __name__ == "__main__":
    from retrieval import retrieve

    query = "What BLEU score did the Transformer (big) model achieve on English-to-German?"
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)
    print("QUERY:", query)
    print("\nANSWER:\n", answer)
