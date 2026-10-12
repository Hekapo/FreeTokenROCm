# Интеграция исследования: этап 1, 12.10.2026

**Статус:** все три изменения реализованы и проверены на CPU. Дополнение после этапа2: GPU sampler/counts/stream gate также пройден, результат в [отчёте этапа2](integration-research-stage2-20261012.md). В самой CPU-итерации этапа1 модель не запускалась.

Ветка: `integration/research-refresh-20261002`, исходный HEAD `e5e2899eef36e55eb84a58a4cd59de8766ac8c46`. Изменения находятся в рабочем дереве. Действующий [план](research-integration-plan-20261012.md).

## Источники и перенос

Ревизии повторно проверены перед реализацией; статусы и SHA совпали с аудитом. Сохранены свежие metadata и исходные файлы до правок.

| Пункт | Источник | Перенесённый контракт |
|---|---|---|
| 1.1 | [FreeToken593](https://github.com/FlashML-org/FreeToken/pull/593), open head `1d5ce5459b6632adc60feb50b07cd4bbf73af1c6`, автор Avicennasis | Проверка FTW output до TP/device setup |
| 1.2 | [FreeToken599](https://github.com/FlashML-org/FreeToken/pull/599), merged `581fdca310d429cca74ac9d94ed2b893b86d9320`, автор yunwei37 | Backend page size в `/v1/stats`, fallback к frontend config |
| 1.3 | [FreeToken527](https://github.com/FlashML-org/FreeToken/pull/527), merged `c8186396f5834994628f7750b11da7300e1d4714`, автор taking-lying-flat | Presence/frequency penalties, request-owned generated-token counts, обновление в sampling stream |

### 1.1. Защита FTW

Область: `checkpoint/convert.py`, `tests/checkpoint/test_convert_refusal.py`, `docs/cli.md`.

- Готовый результат с `freetoken_weight.json` и прерванный с `freetoken-*.ftw` отклоняются с `SystemExit` до настройки TP/устройства и открытия writer.
- Сообщение называет output и найденный marker. Для повторного conversion требуется новый каталог либо явное удаление прежнего результата пользователем. Автоматического удаления и resume нет.
- Отсутствующий/пустой каталог и metadata-only output допустимы; проверка не создаёт и не изменяет файлов. `glob.escape` сохраняет проверку путей с квадратными скобками.
- Маленькие файловые fixtures проверяют bytes и mtime существующих файлов. Полная конвертация checkpoint не выполнялась.

### 1.2. KV statistics

Область: `server/stats.py`, существующий `tests/server/test_generation_accounting.py`.

- Размер страницы берётся из `state.cache_pools`, полученного от backend. Когда metadata отсутствует или page_size равен 0, сохраняется config fallback.
- Used/total pages продолжают приходить из backend replies. Configured context остаётся отдельным полем `model.ctx`, actual prompt tokens — отдельным request accounting. Например, 130 страниц по 64 токена означают capacity 8320, независимо от configured context 65536.
- Проверены пустой cache, page sizes 64/128, изменение capacity и сохранение реального количества prompt tokens. Размеры выделенных pools и API schema не менялись.

### 1.3. Penalties

Область: `core.py`, `engine/sample.py`, `server/api_models.py`, `server/generation.py`, `server/openai_api.py` и соответствующие tests.

- Оба OpenAI endpoints передают параметры через `SamplingParams`/`TokenizeMsg` до sampler.
- HTTP-модели и, дополнительно к donor, прямой конструктор `SamplingParams` отвергают NaN/Inf и значения вне включительного диапазона [-2,2]. Проверка действует и для внутренних клиентов.
- Формула перед greedy или probability sampling: `logit -= frequency * generated_count + presence * (generated_count > 0)`.
- Счётчик принадлежит `Req`, учитывает только generated tokens, выделяется лениво при ненулевом штрафе и decodable request. Prompt и intermediate `ChunkedReq` не заполняют counts. При нулевых штрафах нет дополнительного tensor allocation или копии logits.
- Logits со штрафами переводятся в FP32 и копируются, сохраняя исходный tensor. Mixed greedy rows следуют argmax; новая позиция запроса в batch не меняет его историю.
- Counts обновляются tensor-операцией на sampling stream до следующего sample, без ожидания host ack. `BatchSamplingArgs` удерживает tensor для текущего forward; после освобождения request и последних args владение заканчивается. Глобального словаря по UID/table_idx нет.
- CPU tests используют настоящий PyTorch tensor math/argmax/scatter, но заменяют GPU probability kernel и pinned H2D preparation. Они подтверждают формулу и ownership, а аппаратный stream ordering проверяется отдельно.

## Выполненные проверки

| Проверка | До исправления | После исправления |
|---|---|---|
| FTW output protection | 4 случая occupied output доходили до запрещённого TP/device setup; helper отсутствовал | 5 passed |
| KV backend geometry | 3 failed / 2 passed: backend pages ошибочно отображались как 1 token | Весь затронутый accounting файл: 14 passed |
| Penalties contracts | 52 failed / 1 GPU skipped: поля отсутствовали, API validation отсутствовала, параметры терялись | Sampling/API набор: 91 passed / 1 GPU skipped |
| Совместная CPU приёмка, включая Stop/abort/chunked prefill | Отдельные reproducer выше | **181 passed / 1 skipped / 2 warnings, 7.57 s**, exit 0 |

Первая попытка stats fixture неверно создавала `UserReply` без обязательных `incremental_output`/`finished`; fixture исправлен до внесения production patch. Эта ошибка fixture не используется как доказательство исходного дефекта; валидный before-run сохранён отдельно.

`git diff --check` прошёл; AST parse всех 12 затронутых Python файлов прошёл. Все 18 существовавших dirty файлов, кроме актуализируемого нового плана, сохранены побайтово, включая прежние FLA/diagnostics правки. HEAD и ветка не изменились.

Warnings: установленный Starlette предупреждает о будущем переходе TestClient на httpx2; FLA сообщает о CPU fallback при скрытой видеокарте. Пакеты ради этих предупреждений не менялись.

Среда: CPython 3.12.0; Torch 2.13.0+rocm10.0.0; Triton Windows 3.8.0.post28; Pydantic 2.13.5/core 2.46.5; FastAPI 0.141.1; pytest 9.1.1. Imported package и sampler проверены из текущего checkout. `torch.cuda.is_initialized()` после приёмки — false.

В PowerShell задавались `PYTHONPATH=<checkout>\python`, `PYTHONDONTWRITEBYTECODE=1`, пустые `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `FT_TEST_PENALTY_GPU=0`. Команда из корня checkout:

```powershell
& 'C:\Users\Iskandar\FreeTokenROCm\.venvs\freetoken-rocm1000-triton38-py312\Scripts\python.exe' -X utf8 -B -c 'import torch, pytest, sys; sys.exit(pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider", "tests/checkpoint/test_convert_refusal.py", "tests/server/test_generation_accounting.py", "tests/test_sampling_params.py", "tests/server/test_openai_api.py", "tests/engine/test_sampling_penalties.py", "tests/server/test_prepare_stop_accounting.py", "tests/scheduler/test_req_append_host.py", "tests/scheduler/test_scheduler_chunked_prefill.py", "tests/scheduler/test_abort_inflight_prefill.py"]))'
```

Фактический запуск дополнительно сохранял JUnit XML, проверял imported source/GPU initialization и записывал versions. Точные pytest args и exit code: `cpu-gates.json`.

## Артефакты и откат

Каталог от общего workspace: `rocm10-20260926/research-refresh-20261011-audit/stage1-20261012/`.

- `baseline.json`, `extra-baseline.json`, `control-source/`: исходные hashes/bytes и граница существующих FLA/diagnostics изменений.
- `pr593.json`, `pr599.json`, `pr527.json`: свежие статусы и donor revisions; исходные diff остаются в родительском audit каталоге.
- `convert-before.txt`/`convert-after.txt`, `stats-before-valid.txt`/`stats-after.txt`, `penalties-before.txt`/`penalties-after.txt`: раздельные gates.
- `cpu-gates.txt`, `cpu-gates.json`, `cpu-gates.xml`: совместная приёмка.
- `1.1.patch`, `1.2.patch`, `1.3.patch`: раздельные изменения относительно сохранённых файлов; `verification.json` содержит финальные hashes и preservation check.

Откатывать один пункт по его patch/сохранённым bytes с учётом последующих правок. `git reset --hard` не подходит: в рабочем дереве есть отдельные незакоммиченные FLA/diagnostics изменения. Commit/push не выполнялись.

## Следующее действие

Этап2 выполнен: opt-in GPU fixture `test_gpu_greedy_penalties_and_sampling_stream_order` прошёл в `s2op-20261012-a4` вместе с аппаратным окном GDN; diagnostic и ordinary64K также прошли с clean Stop. Дальнейшая очередь — этап3 действующего плана.

Исправления этапа 1 не квалифицируют новые FLA kernels на GPU и не доказывают устранение прежнего prefill NaN. Скорость измеряется в обычном режиме на предусмотренном этапе, после аппаратной приёмки.
