import os
from typing import List, Dict, Any, Optional
from groq import Groq
from groq import RateLimitError, AuthenticationError, APIConnectionError, APIError

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("llm_service")


class GroqQuotaException(Exception):
    """Raised when Groq rate limit/quota is exceeded."""
    pass


class GroqAuthException(Exception):
    """Raised when Groq API key is invalid."""
    pass


class LLMService:
    """
    Handles prompt formatting and answer generation using dynamic Groq API keys and models,
    with friendly error handling for rate limits and quota exhaustion.
    """

    _instance = None
    _clients: Dict[str, Groq] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance

    def _resolve_groq_key(self, api_key: Optional[str] = None) -> str:
        if api_key and str(api_key).strip():
            return str(api_key).strip()
        try:
            import streamlit as st
            if hasattr(st, "secrets") and len(st.secrets) > 0:
                if "GROQ_API_KEY" in st.secrets:
                    return str(st.secrets["GROQ_API_KEY"]).strip()
                if "groq_api_key" in st.secrets:
                    return str(st.secrets["groq_api_key"]).strip()
                if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
                    return str(st.secrets["groq"]["api_key"]).strip()
        except Exception:
            pass
        return os.environ.get("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", "")

    def _get_client(self, api_key: Optional[str] = None) -> Groq:
        key = self._resolve_groq_key(api_key)
        if not key:
            raise GroqAuthException("No Groq API Key configured.")

        if key not in self._clients:
            logger.info("Initializing new Groq client...")
            self._clients[key] = Groq(api_key=key)
        return self._clients[key]

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

    @staticmethod
    def clean_answer_boilerplates(answer: str, question: str) -> str:
        """
        Removes repetitive self-promotional boilerplate footers like:
        '## How I can help you ...' or 'Feel free to ask more about...'
        unless the user explicitly asked who you are or how you can help.
        """
        if not answer:
            return answer

        import re
        q_lower = question.lower()
        explicit_identity_query = any(w in q_lower for w in ["who are you", "what can you do", "introduce yourself", "how can you help"])

        if not explicit_identity_query:
            # Strip trailing "## How I can help you" sections and subsequent bullet points
            pattern = r"(?:\n\s*---\s*)?\n\s*#{1,4}\s*How I can help you[\s\S]*$"
            cleaned = re.sub(pattern, "", answer, flags=re.IGNORECASE)
            if cleaned.strip():
                answer = cleaned.strip()

            # Also strip trailing "Feel free to ask more about..." if present
            answer = re.sub(r"\n\s*Feel free to ask more about.*$", "", answer, flags=re.IGNORECASE).strip()

        return answer

    def create_chat_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Constructs prompt supporting both grounded document QA and polite, natural Chit Chat.
        """
        if context and context.strip():
            return f"""You are DocuMind AI, an intelligent, polite, and accurate document assistant.

Context from user's uploaded documents:
-------------------------
{context.strip()}
-------------------------

User Message:
{question}

Instructions:
1. If the provided context contains information relevant to the user's question, base your answer on the context and cite the source document name(s) (e.g. `[Source: filename.pdf]`).
2. If the user's question cannot be answered from the provided context (or is a general concept, technical explanation, or greeting), answer the question directly, accurately, and helpfully using your knowledge. Do NOT output negative disclaimers like "The documents you shared do not contain...".
3. Only cite a source document if information from that document was actually used in formulating your response.
4. Format your response cleanly using Markdown (headers, bullet points, bold text).
5. CRITICAL: Answer ONLY the user's question. Do NOT append unnecessary self-introductions, promotional sales pitches, or "How I can help you" sections at the end.

Answer:"""
        else:
            return f"""You are DocuMind AI, a helpful, polite, and intelligent AI assistant.

User Message:
{question}

Instructions:
1. Respond directly, accurately, and thoroughly to the user's question, topic, or greeting.
2. If the question is about a technical topic, tool, or concept, provide a clear, professional explanation focused strictly on that subject.
3. CRITICAL: Do NOT append any self-promotional footer, boilerplate introduction, or "How I can help you" section to your answer unless the user specifically and explicitly asked who you are or what you can do.
4. Format your response cleanly using Markdown (headers, bullet points, bold text).

Answer:"""

    def create_rag_prompt(self, question: str, context: str) -> str:
        """Backward-compatible alias for create_chat_prompt."""
        return self.create_chat_prompt(question, context)

    def generate_answer(
        self,
        question: str,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Calls the Groq chat completion API to generate an answer with automatic rate-limit fallback.
        """
        primary_model = model or settings.GROQ_MODEL
        temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE

        prompt = self.create_chat_prompt(question, context)
        client = self._get_client(api_key)

        models_to_try = [primary_model]
        for fb in ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "groq/compound-mini"]:
            if fb != primary_model and fb not in models_to_try:
                models_to_try.append(fb)

        last_err = None
        for current_model in models_to_try:
            try:
                logger.info(f"Generating answer with model: '{current_model}' (temp: {temp})")
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are DocuMind AI, a helpful, polite, and intelligent AI document assistant capable of both conversational chit-chat and precise document analysis."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temp,
                )
                answer = response.choices[0].message.content.strip()
                answer = self.clean_answer_boilerplates(answer, question)
                logger.info(f"Successfully generated answer from Groq LLM using '{current_model}'.")
                return answer
            except RateLimitError as e:
                logger.warning(f"Groq rate limit hit on '{current_model}': {e}. Attempting next model...")
                last_err = e
                continue
            except AuthenticationError as e:
                logger.error(f"Groq auth error: {e}")
                raise GroqAuthException("Authentication Error: Invalid Groq API Key. Please update your API Settings.")
            except Exception as e:
                logger.error(f"Groq generation error: {e}")
                raise Exception(f"Groq Generation Error: {str(e)}")

        if last_err:
            raise GroqQuotaException(
                "⚠️ Groq Usage Limit Reached\n\n"
                "Your Groq account has reached its current quota or rate limit across all available models.\n\n"
                "Please check your Groq account usage or upgrade your plan.\n\n"
                "Try again in a few seconds."
            )

    def generate_answer_stream(
        self,
        question: str,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Yields streaming chunks from Groq LLM API with automatic rate-limit fallback.
        """
        primary_model = model or settings.GROQ_MODEL
        temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE

        prompt = self.create_chat_prompt(question, context)
        client = self._get_client(api_key)

        models_to_try = [primary_model]
        for fb in ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "groq/compound-mini"]:
            if fb != primary_model and fb not in models_to_try:
                models_to_try.append(fb)


        last_err = None
        for current_model in models_to_try:
            try:
                logger.info(f"Streaming answer with model: '{current_model}' (temp: {temp})")
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are DocuMind AI, a helpful, polite, and intelligent AI document assistant capable of both conversational chit-chat and precise document analysis."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temp,
                    stream=True
                )
                yielded_any = False
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yielded_any = True
                        yield chunk.choices[0].delta.content
                if yielded_any:
                    return
            except RateLimitError as e:
                logger.warning(f"Groq rate limit hit on '{current_model}': {e}. Attempting fallback model...")
                last_err = e
                continue
            except AuthenticationError as e:
                logger.error(f"Groq auth error: {e}")
                raise GroqAuthException("Authentication Error: Invalid Groq API Key. Please update your API Settings.")
            except Exception as e:
                logger.error(f"Groq streaming error: {e}")
                raise Exception(f"Groq Generation Error: {str(e)}")

        if last_err:
            raise GroqQuotaException(
                "⚠️ Groq Usage Limit Reached\n\n"
                "Your Groq account has reached its current quota or rate limit across all available models.\n\n"
                "Please wait a few seconds and try again."
            )