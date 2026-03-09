import argparse

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.lesson_2_3.cli_bot import CliBot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI-скрипт, который возвращает валидный JSON по схеме WeatherInfo.",
    )
    parser.add_argument(
        "city",
        nargs="?",
        help="Название города",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    city = args.city or input("Введите название города: ").strip()

    bot = CliBot()

    try:
        weather_info = bot.get_weather_info(city)
        print(weather_info.model_dump_json(indent=2, ensure_ascii=False))  # noqa: T201
        return 0

    except OutputParserException as exc:
        error_response = bot.build_error(
            message="Invalid response format",
            details=str(exc),
        )
        print(error_response.model_dump_json(indent=2, ensure_ascii=False))  # noqa: T201
        return 3

    except ValidationError as exc:
        error_response = bot.build_error(
            message="Validation failed",
            details=str(exc),
        )
        print(error_response.model_dump_json(indent=2, ensure_ascii=False))  # noqa: T201
        return 4

    except ValueError as exc:
        error_response = bot.build_error(
            message="Invalid input",
            details=str(exc),
        )
        print(error_response.model_dump_json(indent=2, ensure_ascii=False))  # noqa: T201
        return 2

    except Exception as exc:
        error_response = bot.build_error(
            message="Internal error",
            details=str(exc),
        )
        print(error_response.model_dump_json(indent=2, ensure_ascii=False))  # noqa: T201
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
