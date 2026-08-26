# 1. Используем ультра-легкий образ Python
FROM python:3.10-slim

# 2. Устанавливаем рабочую директорию
WORKDIR /app

# 3. Копируем файл зависимостей бота и устанавливаем их
# (Для бота нужны только aiogram, requests/httpx, python-dotenv)
COPY bot/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 4. Копируем исходный код бота
COPY bot/ /app/bot/

# 5. Запускаем бота
CMD ["python", "bot/main.py"]