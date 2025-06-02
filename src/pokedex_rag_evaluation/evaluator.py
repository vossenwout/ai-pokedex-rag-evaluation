import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from google.genai import Client
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from pokedex_rag_evaluation.answer_generator import (
    AnsweredEvaluationSet,
    AnsweredQAPair,
    RetrievedKnowledge,
)
from pokedex_rag_evaluation.metrics import (
    ANSWER_CORRECTNESS,
    CONTEXT_RELEVANCE,
    FAITHFULNESS,
    HELPFULNESS,
    Metric,
    MetricScore,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluatedAnsweredQAPair:
    question: str
    ground_truth: str
    answer: str
    retrieved_knowledge: list[RetrievedKnowledge]
    sources: list[str]
    metrics: list[MetricScore]


@dataclass
class EvaluationReport:
    metadata: dict
    metrics: dict
    evaluated_answered_qa_pairs: list[EvaluatedAnsweredQAPair]


class Evaluator:
    # https://ai.google.dev/gemini-api/docs/rate-limits
    VALID_LLM_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-04-17",
    ]

    def __init__(self, model: str, save_dir: Path):
        self.model = model
        if model not in self.VALID_LLM_MODELS:
            raise ValueError(f"Model {model} is not a valid LLM model.")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("GEMINI_API_KEY env var is not set")
        self.gemini_client = Client(api_key=api_key)
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_hit_rate(self, answered_qa: AnsweredQAPair) -> MetricScore:
        if not answered_qa.sources:
            return MetricScore(score=0.0, reason="No sources provided")

        hits = 0
        retrieved_urls = [
            retrieved_knowledge.url
            for retrieved_knowledge in answered_qa.retrieved_knowledge
        ]
        for source in answered_qa.sources:
            if source in retrieved_urls:
                hits += 1
        return MetricScore(
            score=hits / len(answered_qa.sources),
            reason=f"Hit rate: {hits} / {len(answered_qa.sources)}",
        )

    def _format_genai_metric_prompt(self, answered_qa: AnsweredQAPair, metric: Metric):
        if metric == Metric.ANSWER_CORRECTNESS:
            return ANSWER_CORRECTNESS.prompt.format(
                question=answered_qa.question,
                answer=answered_qa.answer,
                ground_truth=answered_qa.ground_truth,
            )
        elif metric == Metric.FAITHFULNESS:
            return FAITHFULNESS.prompt.format(
                context=answered_qa.retrieved_knowledge,
                answer=answered_qa.answer,
                question=answered_qa.question,
            )
        elif metric == Metric.CONTEXT_RELEVANCE:
            return CONTEXT_RELEVANCE.prompt.format(
                question=answered_qa.question,
                context=answered_qa.retrieved_knowledge,
            )
        elif metric == Metric.HELPFULNESS:
            return HELPFULNESS.prompt.format(
                question=answered_qa.question, answer=answered_qa.answer
            )
        else:
            raise ValueError(f"Metric {metric} is not supported.")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=30, max=60),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_genai_metric_score(self, prompt: str) -> MetricScore:
        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": MetricScore,
            },
        )
        metric_score: MetricScore = response.parsed
        if metric_score.score < 1 or metric_score.score > 5:
            raise ValueError(
                f"Metric score {metric_score.score} is out of range (1-5)."
            )
        return metric_score

    def evaluate(
        self, answered_evaluation_set: AnsweredEvaluationSet, metrics: list[Metric]
    ):
        results = {}

        metrics_results = {metric: [] for metric in metrics}
        evaluated_answered_qa_pairs = []
        for answered_qa in tqdm(
            answered_evaluation_set.answered_qa_pairs,
            desc="Evaluating metrics",
        ):
            qa_pair_metrics = []
            for metric in metrics:
                if metric == Metric.URL_HIT_RATE:
                    metric_score = self._calculate_hit_rate(answered_qa)
                else:
                    prompt = self._format_genai_metric_prompt(answered_qa, metric)
                    try:
                        metric_score = self.get_genai_metric_score(prompt)
                    except Exception as e:
                        logger.error(f"Error getting metric score for {metric}: {e}")
                        continue

                metrics_results[metric].append(metric_score)
                qa_pair_metrics.append(metric_score)

            evaluated_answered_qa_pairs.append(
                EvaluatedAnsweredQAPair(
                    question=answered_qa.question,
                    ground_truth=answered_qa.ground_truth,
                    answer=answered_qa.answer,
                    retrieved_knowledge=answered_qa.retrieved_knowledge,
                    sources=answered_qa.sources,
                    metrics=[metric.model_dump() for metric in qa_pair_metrics],
                )
            )

        for metric in metrics:
            scores = [score.score for score in metrics_results[metric]]
            results[metric.value] = sum(scores) / len(scores) if scores else None
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        metadata = {
            "evaluation_model": self.model,
            "timestamp": timestamp,
        }

        evaluation_report = EvaluationReport(
            metadata=metadata,
            metrics=results,
            evaluated_answered_qa_pairs=evaluated_answered_qa_pairs,
        )

        with open(self.save_dir / f"evaluation_report_{timestamp}.json", "w") as f:
            json.dump(asdict(evaluation_report), f, indent=2, ensure_ascii=False)

        return evaluation_report
