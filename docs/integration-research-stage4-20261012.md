# Интеграция исследования: этап 4, 12.10.2026

**Статус: завершён для текущего Windows/ROCm-профиля.** Выбранные Python-зависимости установлены и проверены в отдельном candidate venv. Контрольная среда сохранена. GGUF-loader адаптирован, ModelScope оформлен опционально. Три коротких запроса модели прошли; модель и наблюдатель остановлены.

Ветка `integration/research-refresh-20261002`, HEAD `e5e2899eef36e55eb84a58a4cd59de8766ac8c46`. Работа остаётся незакоммиченной. Исторический quarantine сохранён; обычный launcher использует контрольную среду.

## 1. Окружения и зависимости

От корня общего workspace `C:\Users\Iskandar\FreeTokenROCm`:

- Control: `.venvs/freetoken-rocm1000-triton38-py312/`.
- Candidate: `.venvs/freetoken-pydeps-20261012-py312/`.
- Candidate создан через `uv venv` на том же CPython3.12.0. 33718 файлов `Lib/site-packages` скопированы независимо, без hardlinks; исходные и скопированные SHA256 совпали. Начальная копия — 9465463545 bytes, 8.815 GiB. Scripts старого venv не копировались.
- Обновления устанавливались только с явным candidate Python. После resolver dry-run скачаны и проверены по publisher SHA256 шесть конкретных wheels; затем установлены локальные файлы с constraints.

| Пакет | Control | Candidate |
|---|---|---|
| Transformers | 5.16.1 | **5.19.0** |
| Hugging Face Hub | 1.29.0 | **1.33.0** |
| FastAPI | 0.141.1 | **0.143.0** |
| Pydantic | 2.13.5 | **2.14.1** |
| Pydantic Core | 2.46.5 | **2.50.1** |
| OpenTelemetry API | отсутствует | **1.44.0**, новая обязательная зависимость FastAPI |

Остальные 68 пакетов сохраняют версии: AMD Torch2.13.0+rocm10.0.0, SDK10.0.0, Triton Windows3.8.0.post28, TVM-FFI0.1.13.post3, OpenAI2.54.0, NumPy2.4.6, flashlib0.3.0 и остальные transitive dependencies. Hub1.33 требует hf-xet>=1.6; Pydantic2.14 требует typing-extensions>=4.16 и typing-inspection>=0.4.4 — эти версии уже были в control.

Полные snapshots: [control constraints](../constraints/windows-rocm1000-control-20261012.txt), [candidate constraints](../constraints/windows-rocm1000-pydeps-20261012.txt), [порядок использования](../constraints/README.md). Все обязательные зависимости текущего `pyproject.toml` удовлетворены. `uv pip check`: **74 packages compatible**. Constraints не заменяют vendor wheel provenance и не выбирают PyPI Torch вместо AMD сборки.

Метаданные конкретных версий перепроверены на PyPI 12.10.2026, включая Windows CPython3.12 wheel Pydantic Core. Transformers5.19 требует Hub>=1.31; выбранный1.33 находится внутри project<2. [Release Transformers5.19](https://github.com/huggingface/transformers/releases/tag/v5.19.0), [Hub1.33](https://github.com/huggingface/huggingface_hub/releases/tag/v1.33.0).

API-группа принята для совместимости актуального набора и изменений validation allocations в Pydantic. Снижение задержки нашего API отдельно не измерялось. В FastAPI0.143 автоматическая настройка exporters стала opt-in; в проверенном запуске OpenTelemetry SDK/exporters не устанавливались и автоматический экспорт не включался. [Pydantic changelog](https://docs.pydantic.dev/latest/changelog/), [FastAPI0.143](https://github.com/fastapi/fastapi/releases/tag/0.143.0).

## 2. Изменения исходников

### GGUF-tokenizer

На новом Transformers исходный loader воспроизводимо падал: `convert_gguf_tokenizer` исчез из `transformers.integrations.ggml`. Новый API находится в `transformers.integrations.gguf`; имена metadata keys также изменились на `tokenizer_type` / `pre_tokenizer_type`. Context7 не дал документации для перемещения, поэтому сверены source в проверенном wheel и [официальный module v5.19](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/integrations/gguf/__init__.py).

Простой перенос import оказался недостаточен: generic converter меняет AddedTokens и Unicode normalization. Словарь/stop IDs и обычные prompt tokens совпали, но строки с combining marks дали другие tokens. Эта смена поведения в рабочий Qwen-профиль не принята.

Для существующего Qwen GGUF маршрута используется публичный `Qwen2Tokenizer` с vocab/merges из metadata и прежними тремя role tokens. Сохранены NFC, split behavior, существующие token IDs и восстановление USER_DEFINED type4 tokens. Generic route поддерживает новый module с copy/rename metadata keys и старый module для старого Transformers; ошибки отсутствующих transitive dependencies не маскируются fallback.

**Точное сравнение real GGUF на control и candidate:** vocab size/IDs/hash, все AddedToken properties/hash, BOS/EOS/stop IDs, chat template, model config, десять Unicode/JSON/whitespace/role-marker cases и два Anthropic/tool prompts совпали. Hash общего контракта: `dbd56d8de3b33378efc512454c3fb0c126af92d4821a06cd256a1d085a27d396`. Frontend count совпал с worker tokenization. Тот же результат подтверждён после изменения source на старом control.

Начальная проверка ошибочно требовала byte-exact encode/decode для NFD текста. На старом tokenizer уже применялся NFC; проверка исправлена на canonical Unicode equivalence с сохранением точных IDs/decoded strings в сравнении окружений. Это дефект тестового условия, не production regression.

### ModelScope

`modelscope>=1.37,<2` перемещён из обязательных зависимостей в `[project.optional-dependencies].modelscope`.

- Remote `--model-source modelscope` импортирует SDK только по необходимости; если самого SDK нет, ошибка предлагает extra либо Hugging Face.
- Локальная директория и локальный GGUF-файл не требуют SDK даже при выборе modelscope source.
- Dummy-weight ignore patterns сохранены. Отсутствующий transitive module SDK не выдаётся за отсутствующий extra.
- [Install docs](install.md) обновлены. Сам ModelScope и модели из сети в этом этапе не скачивались; remote call contract проверен mock SDK.

## 3. Проверки

| Gate | Результат |
|---|---|
| Control source: прежние GGUF fixtures + ModelScope contracts | **9 passed**, 3.77 s |
| Candidate, 10 затронутых test files | **259 passed**, 1 прежнее Starlette/httpx warning, 7.08 s |
| Дополнительные generic converter interfaces/error classification | **3 passed**, 3.70 s |
| Real GGUF tokenizer/control + candidate + control после source patch | Точное совпадение контрактов, GPU не инициализирован, веса не загружены |
| OpenAI/Anthropic validation/dump и четыре JSON schemas | Точное совпадение; hash `0e5825da15fd54f4e9483bf588baaac44509decbf3750c1eec5b0f03169323a9` |
| Resolver / installed Requires-Dist / project requirements | Согласованы; 74-package candidate |
| Model smoke в candidate | **3/3 passed**, без cache resize, clean Stop |

Основной CPU набор включает API streaming/non-streaming, cancellation/accounting, Responses, message wire, maintenance, tokenizer и Qwen GGUF config. Дополнительные три tests проверяют обе generic converter interfaces и transitive error; добавлены после model run, production source затем не менялся.

Legacy fingerprint helper фиксирует только шесть native packages. Для этого запуска добавлен отдельный полный `candidate-packages.json` с SHA в диагностическом плане; перед стартом проверены реальные версии всех 74 пакетов, плюс frozen source/native manifest. Неудачная первоначальная проверка полных версий через native-only manifest остановилась до запуска модели.

### Аппаратный запуск

Run: `rocm10-20260926/results/ctx64k-20261012-142322-954058bc/`.

RX9070XT/gfx1201, Windows11 build26100, прежние AMD Torch/SDK10.0 и driver. Native HIP engine, Qwen3.6-35B-A3B GGUF Q4_1, KV65536 BF16, GDN FP32, prefill512, MoE2048/LRU, serial, concurrency1, graphs0. Отдельные native/JIT caches; native kernels собраны в namespace нового run. Torch-before-Triton/stable knob launcher из прежнего этапа сохранён.

1. OpenAI streaming: prompt26/completion3, `17`, stop, total1.172 s.
2. Anthropic non-streaming с поздней system instruction: count47/actual input47, output3, `17`, end_turn, budget32 соблюдён.
3. OpenAI streaming:26/3, `17`, stop, total0.843 s; content hash совпал с первым ответом.

KV освобождён после каждого запроса, instance_id тот же. Stop: `STOPPED`, root/observer owned false, process_tree_count0, listeners0, recovery_requiredfalse. Supervisor сообщил выход detokenizer после stop-intent14:26:05; нового fault нет.

Перед запуском free commit49.2 GiB; minima RAM26.772 / commit14.929 / local VRAM6.857 GiB. После Stop:52.019 /49.229 /15.070 GiB. Launch threshold45, runtime floors12/4/3 GiB, client budget180 s и request/no-progress45 s сохранены. Длинных запросов и повторной широкой серии не было.

Этот gate квалифицирует короткую совместимость текущего профиля. Нового 64K long benchmark и сравнения4K/8K/64K нет; это этап8. Generic Gemma converter и graphs-on профиль этим model run не квалифицированы.

## 4. Решения по остальным группам, артефакты и откат

- TVM-FFI0.1.14.post1 отложен до этапа5: native ABI/compiler qualification.
- Numba/llvmlite сохранены парой0.67.0/0.49.0. Numba — transitive dependency flashlib; отдельной подтверждённой пользы обновления в этом этапе нет.
- OpenAI3.x, Hub2.x и NumPy2.5.x не включены в этот набор: действующие project bounds прежние.

Audit: `rocm10-20260926/research-refresh-20261011-audit/stage4-20261012/` от общего workspace. Здесь baseline source copies, full control/candidate manifests, независимая copy inventory, PyPI metadata, wheelhouse+publisher hashes, resolver/install logs, CPU XML/logs, tokenizer/API parity data, live plan/spec, control runtime before/after, `acceptance.json`, `verification.json` и раздельные source patches.

Source patches: `4.1.patch` для tokenizer compatibility и dependency snapshots, `4.2.patch` для optional ModelScope. Перед обратным применением сохранить последующие изменения; откат по отдельному diff, без reset всего дерева. Environment rollback выбирает control Python и его snapshot; новый venv не заменяет старый. Для повторного model run нужен свежий source/package manifest и отдельный cache namespace.

**Следующий этап — 5:** отдельный AMD10.1/Torch2.14 candidate с exact wheel provenance и ABI gates. Системный driver/ОС общий для venv; текущую совместимость и обслуживание проверять отдельно по действующему плану. Продолжать в этом чате и существующей ветке.
