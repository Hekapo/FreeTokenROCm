# Интеграция исследования: этап 3, 12.10.2026

**Статус: завершён для текущего Windows/ROCm-профиля.** Интегрированы Anthropic inline system messages и порядок перестроения кэшей. FP32 mRoPE-кандидат неприменим к этой версии исходников. CPU gates и один короткий запуск модели прошли; модель и наблюдатель остановлены.

Ветка `integration/research-refresh-20261002`, HEAD `e5e2899eef36e55eb84a58a4cd59de8766ac8c46`. Предыдущие локальные изменения сохранены. Новых коммитов, push, установок пакетов и изменения обычного профиля в этом этапе нет.

## 1. Источники и решения

Статусы выбранных PR повторно проверены 12.10.2026. Сохранены metadata и исходные diff.

| Источник | Revision | Решение |
|---|---|---|
| [FreeToken608](https://github.com/FlashML-org/FreeToken/pull/608), merged | `93b0a59164bb6d68749298f868ca3738fd464143` | Адаптирован к нашему tokenizer, который возвращает тензоры, и текстовому API без upstream multimodal-полей |
| [FreeToken641](https://github.com/FlashML-org/FreeToken/pull/641), open | `3cae53d415518a53faabac4f099d6be230f63e91` | Не переносить: `_mrope_torch` и `apply_mrope_torch_fallback` отсутствуют локально |
| [FreeToken650](https://github.com/FlashML-org/FreeToken/pull/650), open | `4f4832839719cb814f928821cc40de3336fd4592` | Порядок shrink-before-grow адаптирован к реальным местным pool classes |

### 3.1. Порядок сообщений и tokenizer

- Только ведущие системные сообщения объединяются с верхним `system`. Поздние напоминания сохраняют своё место при преобразовании Anthropic request.
- `TokenizeMsg` и `GenSpec` несут опциональную `inline_system_policy`; старое сообщение без поля декодируется с `None`.
- Новый `--anthropic-inline-system auto|preserve|fold`, default `auto`, доходит до генерации и `/v1/messages/count_tokens`. Для остальных адаптеров политика остаётся `None`.
- `auto` проверяет конкретный renderer: отдельную системную роль, сохранность прежних токенов и reasoning, положение перед началом ответа, накладные токены. Результат кэшируется по tools и template kwargs, до 128 записей.
- `fold` добавляет инструкцию в ближайшее user/tool content либо создаёт user turn. Системная инструкция внутри группы параллельных tool calls ждёт всех результатов; пропущенный результат вызывает `InlineSystemError`.
- В режиме `preserve` без устойчивого ведущего system также используется folding, чтобы tools не привязывались к меняющемуся позднему system.
- Входные сообщения и content blocks при нормализации не мутируются. Ошибка размещения при count_tokens становится HTTP 400.

Сохранён действующий порядок `apply_chat_template(tokenize=False)` и `encode(add_special_tokens=False)`: шаблон сам формирует специальные токены. Это соответствует [документации Transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/chat_templating.md), проверенной через Context7; фактическая совместимость подтверждена на установленном Transformers 5.16.1.

**Проверка настоящего GGUF-tokenizer без модели:** Qwen3.6 выбирает `fold`; обе версии напоминания дают 299 токенов, первые 286 совпадают. Инструкция присутствует один раз после tool result. Два независимых TokenizeManager и frontend count дают одинаковые длины. GPU не инициализирован, веса модели не загружены.

### 3.2. FP32 mRoPE

Донор исправляет запись в alias `.float()` FP32-срезов в двух Torch fallback-функциях. В нашей версии обе функции отсутствуют; `layers/rotary.py` и `kernel/triton/rope.py` предоставляют обычный RoPE. Qwen3.5/3.6 text-only config сводит mRoPE-параметры к partial NeoX RoPE. Воспроизведения этого alias на отсутствующем wrapper нет; новые fallback-функции ради исправления не добавлялись. Это решение о применимости, без заявления, что donor bug исправлен или проверен на нашем GPU.

### 3.3. Перестроение кэшей

- У pool family появился `rebuild_footprint`, использующий её cost model; DSV4 имеет свой override в местном `kvcache/dsv4_paged_pool.py`.
- Геометрии и операции рассчитываются до изменения window override и до разрушения graphs. Затем сначала выполняются уменьшающиеся операции, потом увеличивающиеся; внутри каждой группы сохраняется порядок MoE → KV → GDN.
- Используются существующие in-place rebuild, refresh page table и graph capture. Учитывается дополнительный padding slot GDN.
- Невместимая геометрия отклоняется до освобождения памяти. При allocation failure после teardown исключение и флаг разрушения сохраняются; scheduler по прежней политике завершает работу без автоматического rollback/retry.
- Донорский файл `kvcache/dsv4/v4_pool.py` отсутствует у нас; его sizing logic перенесена в существующий pool.

CPU ledger воспроизводит OOM: MoE 40→60, KV 50→30, cap 100. До исправления увеличение MoE первым требует 110; после исправления peak остаётся 90. Также проверены обратное направление, all-grow/all-shrink, GDN shrink, window-only shrink до MoE grow, порядок расчёта override, rejection до free и отсутствие refresh/capture после OOM. Ссылки pool/page table проверены после успешного resize. Расчёт DSV4 дополнительно сопоставлен с байтами реально выделенных CPU tier tensors, включая explicit window pin.

Это проверка порядка размещения pool buffers; она не обещает ограничить суммарную память всех scratch buffers, allocator cache и внешних приложений.

## 2. Проверки

| Проверка | Результат |
|---|---|
| Исходные регрессии для API/wire/cache | 8 failed / 1 passed до изменений; причины — изменённый prefix, отсутствующая политика и transient OOM |
| Совместный CPU gate, 10 затронутых test files | **243 passed**, 1 прежнее Starlette/httpx warning, 7.57 s |
| Уточнённый DSV4 physical-footprint контракт после общего gate | **1 passed**, 3.97 s; расширен существующий тест |
| Настоящий GGUF-tokenizer, CPU | Passed: fold, count parity, input immutability и 286 общих prefix tokens |
| CLI choices / live prompt / Python и PowerShell syntax | Passed, без модели |
| Три коротких запроса и одно live cache rebuild | **3/3 passed, resize ok**, clean Stop |

Первый baseline collection остановился из-за отсутствовавшего в нашем тестовом файле `pytest` import; import добавлен, затем получено корректное воспроизведение 8 failures. Лог первоначального сбоя сохранён отдельно; он не считается подтверждением production bug.

### Аппаратная проверка

RX 9070 XT / gfx1201, Windows11 build26100, Torch2.13.0+rocm10.0.0 / SDK10.0.0, Triton3.8.0.post28, TVM-FFI0.1.13.post3. Qwen3.6-35B-A3B GGUF Q4_1, BF16 KV, prefill512, concurrency1, graphs0, serial MoE loading, LRU. Временный launcher со стабильным Triton knob из этапа2 сохранён.

Старт: KV65536 / MoE2048. Один idle resize: **KV49152 / MoE2112**, GDN usable slot1 сохранён. После остановки следующий старт по прежнему профилю снова использует KV65536 / MoE2048.

1. OpenAI streaming: 26 prompt / 3 completion, ответ `17`, finish `stop`, total1.094 s.
2. Anthropic non-streaming с поздней system-инструкцией после resize: count **47**, фактический input **47**, output3, ответ `17`, `end_turn`, budget32 соблюдён.
3. OpenAI streaming после resize: 26 / 3, `17`, `stop`, total0.875 s; content hash совпал с первым ответом.

После каждого запроса KV освобождён и instance_id совпал. Stop: `STOPPED`, root/observer owned false, process_tree_count0, listeners0, recovery_requiredfalse. Запись supervisor об остановке detokenizer сделана после `stop-intent`13:14:01; нового runtime fault нет.

Перед запуском свободный commit49.4 GiB. За весь наблюдаемый запуск minima: RAM **27.115**, commit **14.284**, local VRAM **5.501 GiB**. После Stop: RAM52.322 / commit49.857 / VRAM15.104 GiB. Сохранены launch threshold45 и runtime floors12/4/3 GiB. Client budget180 s, request/no-progress45 s, без warmup cycles и длинных промптов.

GPU graph capture включённого профиля и живой DSV4 resize этим запуском не квалифицированы. Исторический quarantine сохранён; обычный release автоматически не продвигался. Эти короткие timings не используются для сравнения скорости4K/8K/64K; сравнение предусмотрено этапом8.

## 3. Артефакты и продолжение

От корня общего workspace:

- `rocm10-20260926/research-refresh-20261011-audit/stage3-20261012/`: baseline copies, metadata/diffs608/641/650, `before.txt`, `cpu-final.xml/txt`, `dsv4-physical-footprint.txt`, `native-tokenizer.json`, `candidate-state.json`, `acceptance.json`, `verification.json`.
- Раздельные изменения: `3.1.patch` для API/tokenizer и `3.3-final.patch` для окончательного cache lifecycle diff; `3.3.patch` сохраняет вариант на момент запуска модели. До любых отмен сохранить последующие изменения и применять только соответствующий обратный patch к проверенной базе; не использовать reset всего рабочего дерева.
- `control-runtime/` и `control-runtime-after/`: снимки внешних PowerShell/client scripts, `control-hashes.json`, строгие `live-plan.json` и `live-spec.json`.
- `rocm10-20260926/results/ctx64k-20261012-131123-a6dcd023/`: run manifest, native/JIT caches, observer, server/client logs, `cache-lifecycle/`, `short-diagnostic-status-after.json`, `resources-after-stop.json`.

Production source совпадает с manifest аппаратного запуска. После него добавлены отчёт/план и уточнены только assertions существующего CPU-теста DSV4; отдельный повтор этого теста прошёл. Пакеты, HEAD и ветка прежние.

**Следующий этап — 4:** выбрать конкретные dependency groups, подготовить отдельные constraints/environment и проверить GGUF/API совместимость. Рабочий ROCm runtime сохраняется контрольным. Продолжать в этом чате и существующей ветке; дополнительная широкая серия модели для завершённого этапа3 не нужна.
