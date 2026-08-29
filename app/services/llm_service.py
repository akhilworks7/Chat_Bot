from typing import List, Dict, Any, Optional
from groq import Groq

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("llm_service")


class LLMService:
    """
    Handles prompt formatting and answer generation using the Groq LLM API.
    """

    _instance = None
    _client = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            logger.info("Initializing Groq client...")
            self._client = Groq(api_key=settings.GROQ_API_KEY)

    def build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved document chunks into structured context for the prompt.
        """
        context_parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.get("source", "Unknown")
            page = doc.get("page", "")
            page_str = f"Page: {page}\n" if page else ""
            content = doc.get("text", "").strip()

            context_parts.append(
                f"Document {i}\nSource: {source}\n{page_str}Content:\n{content}"
            )

        return "\n\n".join(context_parts)

    def create_rag_prompt(self, question: str, context: str) -> str:
        """
        Constructs the strict RAG prompt enforcing grounded generation.
        """
        return f"""You are a helpful, accurate, and context-grounded document assistant.

Your task is to answer the user's question using the information provided in the context below.

Rules:
1. Answer strictly based on the provided context. Do not invent or assume information.
2. If the user asks for a list, summary, or specific details (such as names, advisory board members, advisors, authors, precautions, symptoms, or procedures), extract and present all relevant information found in the context clearly (using bullet points or numbered lists where requested).
3. If only partial information or some members/items are available in the context, provide the items found and state what is mentioned in the context.
4. If the question cannot be answered from the provided context at all, respond with:
   "The information is not available in the provided documents."
5. Format your response cleanly using Markdown (bold text, lists, headers).

Context:
-------------------------
{context}
-------------------------

User Question:
{question}

Answer:"""

    def generate_answer(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Calls the Groq chat completion API to generate an answer based on the provided context.
        """
        model_name = model or settings.GROQ_MODEL
        temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE

        if not context or not context.strip():
            return "The information is not available in the provided documents."

        prompt = self.create_rag_prompt(question, context)

        logger.info(f"Generating answer with model: '{model_name}' (temp: {temp})")

        response = self._client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document QA assistant. Answer only from the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temp,
        )

        answer = response.choices[0].message.content.strip()
        logger.info("Successfully generated answer from Groq LLM.")
        return answer