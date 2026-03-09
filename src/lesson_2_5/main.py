from langchain_openai import ChatOpenAI

from prompts import build_prompt_template
from src.setup.settings import openai_settings


def run() -> None:

    prompt_template = build_prompt_template()

    final_prompt = prompt_template.format(
        user_question="Как удалить дубликаты из списка?"
    )

    llm = ChatOpenAI(
        model=openai_settings.model,
        api_key=openai_settings.key,
        base_url=openai_settings.base_url,
        temperature=0,
        timeout=openai_settings.timeout_seconds,
        max_retries=openai_settings.max_retries,
    )

    response = llm.invoke(final_prompt)

    print(response.content)


if __name__ == "__main__":
    run()