from urllib.parse import urlencode

import requests


def get_dataset_url(public_url: str) -> str:
    """Получает прямую ссылку для загрузки датасета с Яндекс.Диска.

    Args:
        public_url (str): Ссылка на файл датасета.

    Returns:
        str: Прямая ссылка для загрузки (для `wget`).
    """

    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    final_url = base_url + urlencode({"public_key": public_url})

    response = requests.get(final_url)
    return response.json()["href"]


def parse_ingredients(ingredients_id: str, ingredients_map: dict[int, str]) -> str:
    """Преобразует технические идентификаторы ингредиентов в названия продуктов через запятую.

    Args:
        ingredients_id (str): Строка с идентификаторами через `;` вида `ingr_0000000508;ingr_0000000122`
        ingredients_map (dict[int, str]): Словарь для отображений идентификатор->название продукта.

    Returns:
        str: Список продуктов через запятую.
    """

    ids = [int(id.strip().removeprefix("ingr_")) for id in ingredients_id.split(";")]
    return ", ".join(ingredients_map[id] for id in ids)
