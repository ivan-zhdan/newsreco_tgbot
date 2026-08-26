import logging
import asyncio
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import os

# --- Конфигурация бота ---
# Токен бота
API_TOKEN = os.getenv("BOT_TOKEN", "8914864627:AAH71X9L6OrbK4OaYhb3cY3sGXDVDHDx1J8")
# Загружаем URL из переменной окружения (или берем локальный по умолчанию)
API_URL = os.getenv("API_URL", "http://localhost:8000/recommend")

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# --- Вспомогательные функции (моки для теста) ---
# В реальном проекте история пользователя и список новых новостей
# должны браться из базы данных.
def get_user_history(user_id: int):
    """Мок для получения истории пользователя."""
    return [f"N{i}" for i in range(1001, 1006)]


def get_fresh_news_candidates():
    """Мок для получения новых новостей-кандидатов."""
    # Пример формата из вашего news.tsv (ID новости и Категория)
    return [
        {"news_id": "N100", "category": "lifestyle"},
        {"news_id": "N101", "category": "health"},
        {"news_id": "N102", "category": "news"},
        {"news_id": "N103", "category": "sports"},
        {"news_id": "N104", "category": "finance"},
        {"news_id": "N105", "category": "travel"},
    ]


# --- Обработчики команд ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start."""
    welcome_text = (
        f"Привет, {message.from_user.first_name}! 👋\n\n"
        "Я — твой персональный рекомендатель новостей.\n"
        "Основан на модели CatBoost, обученной на датасете MIND.\n\n"
        "Используй команду /recommend, чтобы получить подборку."
    )
    await message.answer(welcome_text)


@dp.message(Command("recommend"))
async def cmd_recommend(message: types.Message):
    """Обработчик команды /recommend."""
    user_id = message.from_user.id

    # 1. Получаем контекст (историю и кандидатов)
    history = get_user_history(user_id)
    candidates = get_fresh_news_candidates()

    # Информируем пользователя
    status_msg = await message.answer("🤔 Собираю лучшие новости для тебя...")

    # 2. Формируем запрос к FastAPI
    request_data = {
        "user_id": str(user_id),
        "history": history,
        "candidates": candidates
    }

    # 3. Делаем запрос к API ранжирования
    try:
        # Используем sync requests внутри async обработчика для простоты,
        # в проде лучше использовать httpx.AsyncClient.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(API_URL, json=request_data, timeout=10))
        response.raise_for_status()  # Проверка ошибок HTTP

        recommendations = response.json()
        ranked_ids = recommendations.get("ranked_news_ids", [])

    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to API: {e}")
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            text="⚠️ Извини, сервис рекомендаций временно недоступен. Попробуй позже."
        )
        return

    # 4. Формируем ответ пользователю
    if not ranked_ids:
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            text="Пока нет новых рекомендаций для тебя."
        )
        return

    # Подготавливаем текст и кнопки
    response_text = "🗞 **Вот твоя персональная подборка:**\n\n"
    keyboard_buttons = []

    for i, news_id in enumerate(ranked_ids):
        # Находим категорию новости для отображения
        category = next((c['category'] for c in candidates if c['news_id'] == news_id), "news")
        response_text += f"{i + 1}. [{category.capitalize()}] Новость {news_id}\n"

        # Добавляем инлайн-кнопку (мок ссылки)
        keyboard_buttons.append(
            [InlineKeyboardButton(text=f"📄 Читать {news_id}", url=f"https://example.com/news/{news_id}")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)

    # Обновляем сообщение со статусом финальным ответом
    await bot.edit_message_text(
        chat_id=message.chat.id,
        message_id=status_msg.message_id,
        text=response_text,
        parse_mode="Markdown",
        reply_markup=keyboard
    )


# Основной цикл запуска бота
async def main():
    logging.info("Starting Telegram Bot...")
    # Удаляем старые вебхуки и пропущенные обновления
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    # Исправление ошибки Event Loop на Windows
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")