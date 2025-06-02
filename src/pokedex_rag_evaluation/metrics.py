from dataclasses import dataclass
from enum import Enum
from textwrap import dedent

from pydantic import BaseModel, conint


@dataclass
class GenAIMetric:
    prompt: str


class MetricScore(BaseModel):
    score: conint(ge=1, le=5)
    reason: str


ANSWER_CORRECTNESS = GenAIMetric(
    prompt=dedent(
        """
        Evaluate the correctness of the following AI-generated answer.

        Question:
        "{question}"

        Answer:
        "{answer}"

        Expected Answer (ground truth):
        "{ground_truth}"

        Is the answer correct compared to the expected answer?
        Return a score between 1 and 5, where 1 is the lowest score and 5 is the highest score.
        Also return the reason for the score.
        """
    )
)

FAITHFULNESS = GenAIMetric(
    prompt=dedent(
        """
        You are evaluating an AI answer for factual grounding.

        Context:
        "{context}"

        Answer:
        "{answer}"

        Question:
        "{question}"

        Is the answer factually supported by the context above?
        Return a score between 1 and 5, where 1 is the lowest score and 5 is the highest score.
        Also return the reason for the score.
        """
    )
)

CONTEXT_RELEVANCE = GenAIMetric(
    prompt=dedent(
        """
        You are evaluating how relevant a context passage is to a user`s question.

        Question:
        "{question}"

        Context:
        "{context}"

        How relevant is the context for answering this question?
        Return a score between 1 and 5, where 1 is the lowest score and 5 is the highest score.
        Also return the reason for the score.
        """
    )
)

HELPFULNESS = GenAIMetric(
    prompt=dedent(
        """
        Evaluate the helpfulness of the following answer to the given question.

        Question:
        "{question}"

        Answer:
        "{answer}"

        Is the answer helpful to the user?
        Return a score between 1 and 5, where 1 is the lowest score and 5 is the highest score.
        Also return the reason for the score.
        """
    )
)


class Metric(Enum):
    ANSWER_CORRECTNESS = "answer_correctness"
    FAITHFULNESS = "faithfulness"
    CONTEXT_RELEVANCE = "context_relevance"
    HELPFULNESS = "helpfulness"
    URL_HIT_RATE = "url_hit_rate"
