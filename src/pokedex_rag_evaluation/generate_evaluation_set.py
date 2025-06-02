import concurrent.futures
import json
import logging
import logging.config
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google.genai import Client
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger(__name__)


def init_config():
    with open("config/logging.yaml") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        logger.info("Logging configured")
    if not load_dotenv("config/.env"):
        raise ValueError("No secrets found in config/.env")
    else:
        logger.info("Secrets loaded from config/.env")


class GeminiQAGenerator:
    class QaPair(BaseModel):
        question: str
        ground_truth: str

    # https://ai.google.dev/gemini-api/docs/rate-limits
    VALID_LLM_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-04-17",
    ]

    def __init__(self, model_name: str, qa_prompt: str | None = None):
        if model_name not in self.VALID_LLM_MODELS:
            raise ValueError(f"Model {model_name} is not a valid LLM model.")
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("GEMINI_API_KEY env var is not set")
        self.gemini_client = Client(api_key=api_key)
        if qa_prompt is not None:
            self.qa_prompt = qa_prompt
        else:
            self.qa_prompt = "Generate a question and answer pair that is about and can be answered with the following knowledge: {knowledge_content}."

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=30, max=60),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def generate_qa_pair(self, knowledge_content: str) -> QaPair:
        reponse = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=self.qa_prompt.format(knowledge_content=knowledge_content),
            config={
                "response_mime_type": "application/json",
                "response_schema": GeminiQAGenerator.QaPair,
            },
        )
        qa_pair: GeminiQAGenerator.QaPair = reponse.parsed
        return qa_pair


@dataclass
class QAPair:
    question: str
    ground_truth: str
    sources: list[str]


@dataclass
class EvaluationSet:
    metadata: dict
    qa_pairs: list[QAPair]


def generate_qa_pair(knowledge_dir: Path, model_name: str) -> QAPair:
    index_md = knowledge_dir / "index.md"
    metadata_json = knowledge_dir / "metadata.json"

    with open(index_md) as f:
        knowledge_content = f.read()

    with open(metadata_json) as f:
        knowledge_metadata = json.load(f)

    gemini_qa_generator = GeminiQAGenerator(model_name=model_name)
    qa_pair = gemini_qa_generator.generate_qa_pair(knowledge_content)

    return QAPair(
        question=qa_pair.question,
        ground_truth=qa_pair.ground_truth,
        sources=[knowledge_metadata["url"]],
    )


def generate_evaluation_set(
    num_questions: int = 10,
    model_name: str = "gemini-2.0-flash",
    knowledge_dir: Path = Path("raw_data"),
    output_dir: Path = Path("evaluation_set"),
    threads: int = 2,
):
    dirs = list(knowledge_dir.glob("**/*"))
    dirs = [d for d in dirs if d.is_dir()]
    selected_dirs = random.sample(dirs, num_questions)
    qa_pairs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_dir = {
            executor.submit(generate_qa_pair, dir, model_name): dir
            for dir in selected_dirs
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_dir),
            total=len(selected_dirs),
            desc="Generating QA pairs",
        ):
            dir = future_to_dir[future]
            try:
                qa_pair = future.result()
                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.error(f"Failed to generate QA pair for {dir}: {e}")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    evaluation_set = EvaluationSet(
        metadata={
            "model": model_name,
            "num_questions": len(qa_pairs),
            "timestamp": timestamp,
        },
        qa_pairs=qa_pairs,
    )
    # save the evaluation set
    with open(output_dir / f"evaluation_set_{timestamp}.json", "w") as f:
        json.dump(asdict(evaluation_set), f, indent=4, ensure_ascii=False)
    logger.info(
        f"Evaluation set saved to {output_dir / f'evaluation_set_{timestamp}.json'}"
    )


if __name__ == "__main__":
    init_config()
    generate_evaluation_set(
        num_questions=10,
        model_name="gemini-2.0-flash",
        knowledge_dir=Path("raw_data"),
        output_dir=Path("evaluation_set"),
        # don't set threads to high if you run into rate limits
        threads=2,
    )
