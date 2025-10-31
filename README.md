# black-white-pic

Мини-приложение на Python, которое:
- забирает случайную картинку из публичного датасета Google QuickDraw (без регистрации),
- получает вместе с картинкой слово-метку (что нарисовано),
- упрощает изображение до чёрно-белой матрицы 7×7 из 0 и 1, где 1 — закрашенный (чёрный) пиксель, 0 — пустой.

Источники данных (стабильные публичные URL):
- Список категорий: `https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt`
- Картинки в формате numpy (по категории): `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{label}.npy`
 - Эмодзи каталог: `https://raw.githubusercontent.com/github/gemoji/master/db/emoji.json`
 - Twemoji PNG (по unified-коду): `https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{unified}.png`
 - Noto Emoji PNG (по unified-коду): `https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u{unified_with_underscores}.png`

Загрузка идёт напрямую по URL, без ключей/регистрации. Выбирается случайная категория и случайный образец.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Emoji (по умолчанию):
```bash
python main.py --print-matrix --size 7x7 --source emoji --emoji-set twemoji
```

QuickDraw (старый режим):
```bash
python main.py --print-matrix --size 7x7 --source quickdraw
```

## HTTP микросервис

Запуск сервера:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Запрос:
```bash
curl "http://localhost:8000/nonogram/create?size=10x10"
```

Ответ JSON:
```json
{
  "emoji": "🙂",
  "label": "smile",
  "matrix": [[0,1,0,...],[...]]
}
```

Параметры запроса:
- `size` (например `10x10`) — размер выходной матрицы
- `emoji_set` (`twemoji`|`noto`, по умолчанию `twemoji`)
- `label_mode` (`alias`|`emoji`, по умолчанию `alias`)
- `method` (`avg`|`max`|`fraction`, по умолчанию `fraction`)
- `fraction` (0..1, по умолчанию 0.25)
- `threshold` (0..255, по умолчанию 128)
- `threshold_mode` (`global`|`otsu`, по умолчанию `otsu`)
- `autocontrast` (bool, по умолчанию `true`)
- `gamma` (float, по умолчанию 1.0)
- `blur` (float, по умолчанию 0.4)
- `emoji_fetch_retries` (int, по умолчанию 5)
- `emoji_catalog_url` (повторяемый, чтобы переопределить URL каталога эмодзи)
- `custom_emoji_url` (повторяемый, собственные URL картинок)

### Аутентификация

POST `/auth/login` — проверка username и password в БД:

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "player1", "password": "password123"}'
```

Успешный ответ (200):
```json
{
  "success": true,
  "user": {
    "id": 1,
    "username": "player1",
    "is_admin": true
  }
}
```

Ошибка при неверных данных (401):
```json
{
  "detail": "Invalid username or password"
}
```

**Настройка подключения к БД**: Параметры подключения хранятся в `db_config.py` или могут быть переопределены через переменные окружения:
- `DB_HOST` (по умолчанию: localhost)
- `DB_PORT` (по умолчанию: 5432)
- `DB_NAME` (по умолчанию: nonogram_db)
- `DB_USER` (по умолчанию: postgres)
- `DB_PASSWORD` (по умолчанию: пустая строка)

## База данных PostgreSQL

Инициализация БД:

```bash
# Создайте БД PostgreSQL
createdb nonogram_db

# Примените скрипт инициализации
psql -d nonogram_db -f init_db.sql
```

Или с указанием пользователя и хоста:
```bash
psql -h localhost -U postgres -d nonogram_db -f init_db.sql
```

Скрипт создаёт:
- **users** — пользователи (2 примера: player1/password123, player2/secret456)
- **difficulty** — уровни сложности (easy, medium, hard)
- **levels** — уровни игры (5 easy, 10 medium, 15 hard)
- **user_progress** — прогресс пользователей по уровням

**Важно**: В продакшене пароли должны быть захешированы (например, с помощью bcrypt). В примере используются plaintext пароли только для тестирования.

Пример вывода:

```json
{"label": "cat", "matrix": [[0,0,0,1,0,0,0], [0,0,1,1,1,0,0], [0,1,0,1,0,1,0], [0,1,1,1,1,1,0], [0,0,0,1,0,0,0], [0,0,1,1,1,0,0], [0,1,0,0,0,1,0]]}

7x7 matrix:
0 0 0 1 0 0 0
0 0 1 1 1 0 0
0 1 0 1 0 1 0
0 1 1 1 1 1 0
0 0 0 1 0 0 0
0 0 1 1 1 0 0
0 1 0 0 0 1 0
```

## Опции

- `--size WIDTHxHEIGHT` (по умолчанию `7x7`): размер выходной матрицы, например `10x12`.
- `--threshold` (по умолчанию 128): пиксели >= threshold считаются чёрными (`1`).
- `--invert`: инвертировать яркость перед порогом (если кажется, что 0/1 перевёрнуты).
- `--print-matrix`: дополнительно печатать матрицу 7×7 в удобном виде.
- `--style {blocks,ascii,numbers}`: стиль визуализации матрицы.
- `--save-original PATH`: сохранить исходное изображение 28×28 в PNG по указанному пути.
 - `--save-raw PATH`: сохранить исходное исходное изображение (например, PNG эмодзи) без изменений.
 - `--source {emoji,quickdraw}` (по умолчанию `emoji`): источник данных.
 - `--emoji-set {twemoji,noto}` (по умолчанию `twemoji`): набор эмодзи для загрузки.
 - `--custom-emoji-url URL` (можно повторять): использовать список кастомных эмодзи (из этого списка выбирается случайно).
 - `--method {avg,max,fraction}` (по умолчанию `avg`):
   - `avg` — усреднение по площади (как раньше), затем порог.
   - `max` — если в «ячейке» есть хоть один штрих, пиксель=1 (лучше сохраняет тонкие линии при маленьких размерах).
   - `fraction` — пиксель=1, если доля штрихов в ячейке ≥ `--fraction`.
 - `--fraction FLOAT` (по умолчанию `0.2`): порог доли для `method=fraction` (0..1).

## Технические детали

- Модуль `quickdraw_fetcher.py` выбирает категорию и забирает .npy с массивом формата `(N, 784)` и возвращает случайное изображение `(28, 28)` и метку.
- Модуль `processor.py` уменьшает до `7x7` методом `BOX` (сохранение присутствия штрихов) и биноризует матрицу до 0/1.
