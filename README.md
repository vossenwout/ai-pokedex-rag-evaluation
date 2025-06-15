# AI Pokedex Evaluation Framework

This project contains a framework to evaluate our AI Pokedex Assistant. It contains a script to generate evaluation data as well as a script to evaluate the answers of the assistant different metrics.

<img src="assets/banner.png" alt="Pokedex Frontend Screenshot" width="300"/>

## AI Pokedex Project Repos

These are the repos that are used to create and run the AI Pokedex.

- [Knowledgebase and Scraper](https://github.com/vossenwout/pokedex-scraper)
- [Assistant API](https://github.com/vossenwout/pokedex-rag-api)
- [Frontend](https://github.com/vossenwout/pokedex-frontend)
- [Evaluation Framework](https://github.com/vossenwout/pokedex-rag-evaluation)

I created a youtube video where I explain the project: https://www.youtube.com/watch?v=dQw4w9WgXcQ

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management. You can install it with the following commands:
  - **macOS, Linux, or WSL:**
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
  - **Windows (PowerShell):**
    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/vossenwout/pokedex-rag-evaluation
    cd pokedex-rag-evaluation
    ```

2.  **Install dependencies:**
    Use Poetry to install the required Python packages.

    ```bash
    poetry install
    ```

## Configuration

To run the evaluation framework you are required to create a `.env` file in the config folder of the project.

```bash
touch config/.env
```

This env file should contain the env vars of `config/.env.example`.

| Environment Variable             | Description                         | Where to get it              |
| -------------------------------- | ----------------------------------- | ---------------------------- |
| `GEMINI_API_KEY`                 | API key for accessing Gemini models | https://aistudio.google.com/ |
| `ZILLIZ_CLUSTER_PUBLIC_ENDPOINT` | Public endpoint for Zilliz cluster  | https://zilliz.com/          |
| `ZILLIZ_CLUSTER_TOKEN`           | Authentication token for Zilliz     | https://zilliz.com/          |

## Generating Evaluation Sets

By leveraging Gemini models we can generate evaluation sets based on the data we scraped.

1. **Eveluation set generation**:
   Create a `raw/` folder in which you manually copy the raw data scraped by the [Scraper](https://github.com/vossenwout/pokedex-scraper).

2. **Run the evaluation set generation script:**

   ```bash
   poetry run python src/pokedex_rag_evaluation/generate_evaluation_set.py
   ```

3. **Results**

   Evaluation sets are stored in the `/evaluation_set` folder.

## Evaluation the assistant

Evaluation Metrics:

- `ANSWER_CORRECTNESS`: Evaluates the correctness of the answer.
- `FAITHFULNESS`: Evaluates the groundedness of the answer.
- `CONTEXT_RELEVANCE`: Evaluates the relevance of the retrieved context.
- `HELPFULNESS`: Evaluates the helpfulness of the answer.
- `URL_HIT_RATE`: Evaluates whether we retrieve the correct urls.

After generating evaluation sets you can evaluate the assistant by running the following command:

1. **Run the evaluation script:**

   ```bash
   poetry run python -m pokedex_rag_evaluation
   ```

2. **Results**

   Evaluation results are stored in the `/results` folder.

## Project Structure

- `evaluation_set/`: Contains generated evaluation sets.
- `results/`: Contains evaluation results.
- `src/pokedex_rag_evaluation/generate_evaluation_set.py`: Contains the script to generate evaluation sets.
- `src/pokedex_rag_evaluation/__main__.py`: Contains the script to evaluate the assistant.
- `src/pokedex_rag_evaluation/metrics.py`: Contains the implemented GenAI metrics to evaluate the assistant.
