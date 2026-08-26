# 1. Используем официальный легкий образ Python 3.10
FROM python:3.10-slim

# 2. Устанавливаем системные зависимости для CatBoost и корректной сборки пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 4. Копируем requirements.txt из папки src/ и устанавливаем зависимости
# Оптимизирует кэширование Docker: слой пересобирается только при изменении зависимостей
COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 5. Копируем исходный код приложения и веса модели в контейнер
COPY src/ /app/src/
COPY models/ /app/models/

# 6. Открываем порт 8000 для подключения FastAPI
EXPOSE 8000

# 7. Указываем команду для запуска сервиса из рабочей директории
CMD ["python", "src/app.py"]