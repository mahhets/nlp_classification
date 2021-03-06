# NLP_classification

API, способное классифицировать русскоязычные сообщения по двум тематикам:
- Python
- DataScience

# Research

- Все исследования находятся в ноутбуке src\research.ipynb

# Установка, настройка и запуск микросервиса

## 1. Установка
- Для работы микросервиса необходим `python3.8`

### Порядок установки:
1. Клонирование репозитория
```shell
git clone <путь к удаленному репозиторию>
```

2. Установка и активация виртуального окружения

Находясь в корне клонированного каталога, выполнить:

```shell
python -m venv <название>
. venv/bin/activate
```

3. Установка зависимостей в виртуальное окружение:

Находясь в корне клонированного каталога, выполнить:
```shell
pip install -r requirements.txt
```
PS. Если используется только микросервис(без Jupyter) - установить зависимости из requirements_api.txt

## 2. Настройка микросервиса

- Файл `env.example` в корне каталога - это пример файла `.env`, который необходимо создать в /src и из которого приложение читает:
P.S. Я оставлю .env в src/ для удобства

1. `SERVER HOST` и `SERVER PORT` - адрес и порт сервера, на котором микросервис будет слушать запросы
2. Остальные настройки оставить без изменений

## 3. Обучение модели

### Для дальнейшей настройки необходимо, находясь в корне каталога, перейти в src/nlp_classification/

```shell
cd src/nlp_classification/
```

- src/nlp_classification/get_model.py  - скрипт оборачивает модель в пайплайн, обучает и сохраняет. Результатом выполнения будет сериализированны пайплайн со всеми обработками.

Для обучения и сохранения модели необходимо выполнить инструкцию ниже:

```shell
python3 get_model.py 
```

## 4. Запуск сервера

Запуск сервиса необходимо производить из src/nlp_classification/. Для запуска выполнить инструкцию ниже:

```shell
python3 web_server.py
```


# Как пользоваться

## На вход сервису подаются:

- `text` - Обычный текст для классификации. Пример: В питоне очень приятный синтаксис

## Пример запроса
```
server_adress/predict?text=В питоне очень приятный синтаксис
```

## Сервис возвращает:

- Сервис возвращает название чата, к которому ML-модель отнесла данное сообщение

Пример ответа для запроса выше:

- {"Chat":"Python"}




