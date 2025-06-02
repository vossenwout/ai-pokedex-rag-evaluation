import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from rag_assistant.assistant import ConversationTurn, PokedexAssistant
from rag_assistant.assistant_config import load_assistant_config
from rag_assistant.knowledgebase import MilvusKnowledgeBase
from tqdm import tqdm

from pokedex_rag_evaluation.generate_evaluation_set import EvaluationSet, QAPair

logger = logging.getLogger(__name__)


@dataclass
class RetrievedKnowledge:
    text: str
    url: str


@dataclass
class AnsweredQAPair:
    question: str
    ground_truth: str
    answer: str
    retrieved_knowledge: list[RetrievedKnowledge]
    sources: list[str]


@dataclass
class AnsweredEvaluationSet:
    metadata: dict
    answered_qa_pairs: list[AnsweredQAPair]


class AnswerGenerator:
    def __init__(self, assistant_config_path: Path):
        config = load_assistant_config(assistant_config_path)

        self.assistant_to_test = PokedexAssistant(
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

    def complete_evaluation_set(
        self, evaluation_set_path: Path
    ) -> AnsweredEvaluationSet:
        with open(evaluation_set_path) as f:
            raw = yaml.safe_load(f)
            evaluation_set = EvaluationSet(
                metadata=raw["metadata"],
                qa_pairs=[QAPair(**qa) for qa in raw["qa_pairs"]],
            )
        answered_qa_pairs = []
        for qa_pair in tqdm(
            evaluation_set.qa_pairs,
            total=len(evaluation_set.qa_pairs),
            desc="Answering questions",
        ):
            messages = [
                ConversationTurn(role="user", content=qa_pair.question),
            ]
            try:
                response, retrieved_knowledge = self.assistant_to_test.chat(messages)
            except Exception as e:
                logger.error(f"Failed to generate answer for {qa_pair.question}: {e}")
                continue
            answered_qa_pairs.append(
                AnsweredQAPair(
                    question=qa_pair.question,
                    ground_truth=qa_pair.ground_truth,
                    answer=response.content,
                    retrieved_knowledge=[
                        RetrievedKnowledge(
                            text=result.text,
                            url=result.metadata.url,
                        )
                        for result in retrieved_knowledge
                    ],
                    sources=qa_pair.sources,
                )
            )
        return AnsweredEvaluationSet(
            metadata=evaluation_set.metadata,
            answered_qa_pairs=answered_qa_pairs,
        )
