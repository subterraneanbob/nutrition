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
