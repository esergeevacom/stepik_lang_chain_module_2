import re

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.setup.settings import openai_settings

load_dotenv()


class WeatherInfo(BaseModel):
    city: str = Field(description="Город")
    temperature: float = Field(description="Температура")
    condition: str = Field(description="Условия")


class ErrorResponse(BaseModel):
    error: str
    details: str | None = None


class CliBot:

    def __init__(self, model_name: str | None = None) -> None:
        # Создаём парсер
        self.output_parser = PydanticOutputParser(pydantic_object=WeatherInfo)

        # Создаем шаблон промпта
        self.prompt = self._get_prompt_template()

        # Создаём модель
        self.llm = ChatOpenAI(
            model=model_name or openai_settings.model,
            api_key=openai_settings.key,
            base_url=openai_settings.base_url,
            temperature=openai_settings.temperature,
            timeout=openai_settings.timeout_seconds,
            max_retries=openai_settings.max_retries,
        )

        self.chain = self.prompt | self.llm

    def _get_prompt_template(self) -> PromptTemplate:
        """Шаблон промпта."""
        return PromptTemplate(
            template=(
                "Верни только валидный json без markdown, комментариев и пояснений.\n"
                "Не возвращай json schema, не возвращай поля вроде properties, required, type.\n"
                "Нужен экземпляр данных, который соответствует этой схеме:\n\n"
                "{format_instructions}\n\n"
                "Используй входные данные пользователя.\n"
                "Запрос: {city}\n"
                "Ответ:"
            ),
            input_variables=["city"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

    @staticmethod
    def _extract_json_object(raw_text: str) -> str:
        cleaned_text = raw_text.strip()

        markdown_match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned_text, re.DOTALL)
        if markdown_match:
            return markdown_match.group(1)

        object_match = re.search(r"(\{.*\})", cleaned_text, re.DOTALL)
        if object_match:
            return object_match.group(1)

        return cleaned_text

    def get_weather_info(self, city: str) -> WeatherInfo:
        normalized_city = city.strip()
        if not normalized_city:
            raise ValueError("City name must not be empty")

        response = self.chain.invoke({"city": normalized_city})
        raw_response = response.content if hasattr(response, "content") else str(response)
        extracted_json = self._extract_json_object(raw_response)

        try:
            return self.output_parser.parse(extracted_json)
        except OutputParserException:
            repair_prompt = (
                "Преобразуй следующий ответ в строго валидный json "
                "по указанной схеме. Верни только json и ничего больше.\n\n"
                f"{self.output_parser.get_format_instructions()}\n\n"
                f"Исходный ответ:\n{raw_response}"
            )
            repaired_response = self.llm.invoke(repair_prompt)
            repaired_text = (
                repaired_response.content
                if hasattr(repaired_response, "content")
                else str(repaired_response)
            )
            repaired_json = self._extract_json_object(str(repaired_text))

            return self.output_parser.parse(repaired_json)

    @staticmethod
    def build_error(
        message: str,
        details: str | None = None,
    ) -> ErrorResponse:
        return ErrorResponse(error=message, details=details)
