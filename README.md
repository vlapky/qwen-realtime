# Qwen Image 2.1 Realtime: локальный UI + инференс на RunPod Serverless

Копия спейса [linoyts/qwen-image-2-1-realtime](https://huggingface.co/spaces/linoyts/qwen-image-2-1-realtime).
Модель та же: Qwen-Image-2.1 + 2-step DMD2 LoRA (вшита в веса), sigmas [1.0, 0.5], CFG 1, AOTI-ядра.

- `Dockerfile` + `handler.py` — воркер RunPod, делает только инференс. Базовая модель (~33 ГБ)
  не входит в образ: RunPod заранее кладёт её на хост через **Cached models**, время скачивания не тарифицируется.
  LoRA и AOTI-ядра (~76 МБ) вшиты в образ. Модель загружается и прогревается один раз при старте воркера.
- `qwen21_aoti.py` — AOTI-ядра из спейса без пакета `spaces`. Работают только на H100/H200
  (sm_90), на других GPU воркер сам переходит в обычный режим (eager).
- `app.py` — локальный Gradio-интерфейс. Режим **⚡ 2-step реалтайм** (512px) или **Обычный**
  (512 / 1K / 1.5K). Запрос в API уходит только по кнопке Generate.

## 1. Залить код на GitHub

RunPod собирает образ сам из GitHub-репозитория, локально Docker не нужен.

```bash
git add -A && git commit -m "Qwen Image 2.1 RunPod worker" && git push
```

## 2. Создать endpoint в консоли RunPod

1. Settings → Connections → GitHub → Connect (дать доступ к репозиторию).
2. Serverless → New Endpoint → Import Git Repository → выбрать репозиторий, ветку, Dockerfile Path: `Dockerfile`.
3. Endpoint Type: **Queue**.
4. GPU: **H100 80GB** (или H200). На 48 ГБ (L40S / A6000 / RTX 6000 Ada) модель влезет, но без AOTI будет медленнее.
5. **Model** (Cached models): `Qwen/Qwen-Image-2.1`.
6. Advanced → Allowed CUDA versions: **13.0** и новее (torch 2.13+cu130).
7. Холодный старт:
   - FlashBoot: включён.
   - Idle timeout: 60–300 с, чтобы воркер оставался тёплым между запросами.
   - Active workers: 1, если нужен совсем без холодного старта (тарифицируется постоянно).
   - Execution timeout: 600 с.
8. Deploy. Сборка видна на вкладке Builds. Обновление образа — через новый GitHub release.

## 3. Запустить интерфейс

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
```

```bash
export RUNPOD_API_KEY=ваш_ключ RUNPOD_ENDPOINT_ID=id_endpoint
```

```bash
.venv/bin/python app.py
```

Откроется http://127.0.0.1:7860. Под картинкой видно время на GPU, полное время и сколько ушло на
очередь или холодный старт.

## Настройки

Переменные окружения endpoint'а:

| Переменная | По умолчанию | Что делает |
| --- | --- | --- |
| `QWEN21_AOTI` | `1` | `0` — отключить AOTI-ядра |

Переменные окружения для `app.py`:

| Переменная | Что делает |
| --- | --- |
| `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID` | Доступ к endpoint |
| `RUNPOD_TIMEOUT` | Сколько ждать ответа, по умолчанию 900 с (с запасом на холодный старт) |
| `HF_TOKEN` | Для «Enhance prompt» через HF-спейс `hugging-apps/qwen-image-2-1-prompt-enhancer` |
