import logging
import logging.config
import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import GenerateContentConfig, ModelContent, Part, UserContent
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from rag_assistant.assistant_config import load_assistant_config
from rag_assistant.knowledgebase import MilvusKnowledgeBase, PokedexSearchResult

logger = logging.getLogger(__name__)

# https://ai.google.dev/gemini-api/docs/rate-limits
VALID_LLM_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-04-17",
]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GenerateRetrievalQueryError(Exception):
    pass


class GenerateAnswerError(Exception):
    pass


class PokedexAssistant:
    def __init__(
        self,
        answer_generation_prompt: str,
        retrieval_query_prompt: str,
        top_p: float,
        knowledge_base: MilvusKnowledgeBase,
        answer_generation_model: str,
        retrieval_query_model: str,
    ):
        self.answer_generation_prompt = answer_generation_prompt
        self.retrieval_query_prompt = retrieval_query_prompt
        self.top_p = top_p
        self.knowledge_base = knowledge_base
        if answer_generation_model not in VALID_LLM_MODELS:
            raise ValueError(
                f"Invalid LLM: {answer_generation_model}. Valid LLMs: {VALID_LLM_MODELS}"
            )
        self.answer_generation_model = answer_generation_model
        if retrieval_query_model not in VALID_LLM_MODELS:
            raise ValueError(
                f"Invalid retrieval query LLM: {retrieval_query_model}. Valid LLMs: {VALID_LLM_MODELS}"
            )
        self.retrieval_query_model = retrieval_query_model
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("GEMINI_API_KEY env var is not set")
        self.gemini_client = Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=30, max=60),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_retrieval_query(self, messages: list[ConversationTurn]) -> str:
        return self.gemini_client.models.generate_content(
            model=self.retrieval_query_model,
            config=GenerateContentConfig(
                system_instruction=self.retrieval_query_prompt,
            ),
            contents=[
                UserContent(
                    parts=[
                        Part(
                            text="Generate a retrieval query for the following conversation: "
                        ),
                        Part(
                            text="\n\n".join(
                                [f"{turn.role}: {turn.content}" for turn in messages]
                            )
                        ),
                    ],
                ),
            ],
        ).text

    def _construct_gemini_rag_prompt(
        self,
        messages: list[ConversationTurn],
        retrieved_chunks: list[PokedexSearchResult],
    ) -> str:
        contents = []
        for message in messages:
            if message.role == "user":
                contents.append(UserContent(parts=[Part(text=message.content)]))
            else:
                contents.append(ModelContent(parts=[Part(text=message.content)]))
        sources_string = "\n\n".join(
            [
                f"--- Source {i+1} ---\nURL: {result.metadata.url}\nText: {result.text}"
                for i, result in enumerate(retrieved_chunks)
            ]
        )
        sources_string = "Retrieved sources:\n\n" + sources_string
        contents.append(UserContent(parts=[Part(text=sources_string)]))
        return contents

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=30, max=60),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_answer(
        self,
        messages: list[ConversationTurn],
        retrieved_chunks: list[PokedexSearchResult],
    ) -> str:
        return self.gemini_client.models.generate_content(
            model=self.answer_generation_model,
            config=GenerateContentConfig(
                system_instruction=self.answer_generation_prompt,
                top_p=self.top_p,
            ),
            contents=self._construct_gemini_rag_prompt(messages, retrieved_chunks),
        ).text

    def chat(
        self, messages: list[ConversationTurn]
    ) -> tuple[ConversationTurn, list[PokedexSearchResult]]:
        if not messages or messages[-1].role != "user":
            raise ValueError("Messages must end with a user message")

        if len(messages) > 1:
            try:
                retrieval_query = self._generate_retrieval_query(messages)
            except Exception as e:
                raise GenerateRetrievalQueryError(
                    f"Failed to generate retrieval query: {e}"
                ) from e
        else:
            retrieval_query = messages[-1].content

        retrieved_knowledge = self.knowledge_base.hybrid_search(retrieval_query)

        try:
            answer = self._generate_answer(messages, retrieved_knowledge)
        except Exception as e:
            raise GenerateAnswerError(f"Failed to generate answer: {e}") from e

        return (
            ConversationTurn(
                role="assistant",
                content=answer,
            ),
            retrieved_knowledge,
        )


if __name__ == "__main__":
    load_dotenv("config/.env")
    with open("config/logging.yaml") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        logger.info("Logging configured")

    config = load_assistant_config("config/assistant.yaml")

    assistant = PokedexAssistant(
        answer_generation_prompt=config.answer_generation_prompt,
        retrieval_query_prompt=config.retrieval_query_prompt,
        top_p=config.top_p,
        knowledge_base=MilvusKnowledgeBase(
            milvus_collection_name=config.knowledge_base.collection_name,
            embedding_model=config.knowledge_base.embedding_model,
            search_limit=config.knowledge_base.search_limit,
        ),
        answer_generation_model=config.answer_generation_model,
        retrieval_query_model=config.retrieval_query_model,
    )

    messages = [
        ConversationTurn(role="user", content="What is the type of Charmander?"),
    ]
    response, _ = assistant.chat(messages)
    messages.append(response)
    logger.info(response.content)
    new_question = "What is the evolution?"
    messages.append(ConversationTurn(role="user", content=new_question))
    response, _ = assistant.chat(messages)
    messages.append(response)
    logger.info(response.content)
    new_question = "What is a polimerization?"
    messages.append(ConversationTurn(role="user", content=new_question))
    response, _ = assistant.chat(messages)
    messages.append(response)
    logger.info(response.content)
