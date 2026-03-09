from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate


class FewShotExample(BaseModel):
    question: str
    answer: str


class FewShotExamplesFile(BaseModel):
    examples: list[FewShotExample] = Field(default_factory=list)


def escape_template_text(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def load_examples(file_path: str) -> list[dict[str, str]]:
    file_content = Path(file_path).read_text(encoding="utf-8")
    raw_data = yaml.safe_load(file_content)

    parsed_examples = FewShotExamplesFile.model_validate(raw_data)

    return [
        {
            "question": escape_template_text(example.question),
            "answer": escape_template_text(example.answer),
        }
        for example in parsed_examples.examples
    ]


def build_prompt_template(examples_file_path: str = "examples.yaml") -> FewShotPromptTemplate:
    examples = load_examples(examples_file_path)

    example_prompt = PromptTemplate.from_template(
        "Вопрос пользователя:\n{question}\n\nОтвет ассистента:\n{answer}"
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=(
            "Ты помощник по Python.\n"
            "Отвечай на русском языке.\n"
            "Всегда:\n"
            "1. Давай краткое объяснение.\n"
            "2. Показывай пример Python-кода.\n"
            "3. Поясняй, почему решение работает.\n"
        ),
        suffix="Вопрос пользователя:\n{user_question}\n\nОтвет ассистента:\n",
        input_variables=["user_question"],
        example_separator="\n\n",
    )

    return few_shot_prompt