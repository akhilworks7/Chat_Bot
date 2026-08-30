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

    def _get_client(self, api_key: Optional[str] = None) -> Groq:
        key = api_key or settings.GROQ_API_KEY
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
1. If the user's message is asking about information in their uploaded documents, answer accurately and factually using the provided context. If the context contains partial information, share what is mentioned.
2. If the user's message is a greeting, polite pleasantry (e.g., 'hi', 'hello', 'how are you', 'thank you'), casual conversation, or general query not covered in the context, respond politely, naturally, and helpfully as DocuMind AI.
3. If the user asks a question expecting document-specific facts that are completely absent from the context, politely clarify that the provided documents do not contain that information, but offer to help with general questions or other topics.
4. Format your response cleanly using Markdown (headers, bullet points, bold text).

Answer:"""
        else:
            return f"""You are DocuMind AI, a friendly, polite, and intelligent document assistant and conversational AI.

User Message:
{question}

Instructions:
1. Respond politely, naturally, and helpfully to the user's greeting, question, or casual conversation.
2. If the user asks what you can do or how to use the app, explain that you are DocuMind AI and can answer general questions as well as ingest and analyze PDF documents uploaded to their DocuMind workspace.
3. Format your response cleanly using Markdown (headers, bullet points, bold text).

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