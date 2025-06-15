import logging
from pathlib import Path

from pokedex_rag_evaluation.answer_generator import AnswerGenerator
from pokedex_rag_evaluation.evaluator import Evaluator
from pokedex_rag_evaluation.metrics import Metric

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    answer_generator = AnswerGenerator(
        assistant_config_path=Path("config/assistants/assistant_v1.yaml")
    )
    answered_evaluation_set = answer_generator.complete_evaluation_set(
        evaluation_set_path=Path(
            "evaluation_set/evaluation_set_2025_06_14_11_57_23.json"
        )
    )
    evaluator = Evaluator(
        model="gemini-2.0-flash-lite",
        save_dir=Path("results"),
    )
    evaluator.evaluate(
        answered_evaluation_set=answered_evaluation_set,
        metrics=[
            Metric.URL_HIT_RATE,
            Metric.FAITHFULNESS,
            Metric.CONTEXT_RELEVANCE,
            Metric.HELPFULNESS,
        ],
    )
    logger.info("Evaluation set completed successfully.")
