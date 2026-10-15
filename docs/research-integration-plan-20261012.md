# План интеграции результатов исследования: актуализация 15.10.2026

**Действующий статус по сохранённым данным:** этапы 0–4, функциональное
применение ROCm 10.1 с compiler workaround, штатный launcher 64K и backend
CPU affinity завершены. Изменения слиты и отправлены в пользовательский
`fork/main`; база текущей работы — `dac3d655`. Полная серия на новом SDK
дала 353 корректных запроса и чистый Stop. Ускорение самого SDK не подтверждено.

В `integration/backend-progress-diagnostics` подготовлены локальные изменения
stack/stage capture; сохранены 118 успешных CPU-проверок. Модель с этой
диагностикой ещё не запускалась, изменения не закоммичены. Сводный отчёт по
логам готов. Основной источник актуализации —
[сводка от 15 октября](C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/64k-saved-logs-summary-20261015.md).

**Текущий режим по просьбе пользователя:** работа с планом и документами,
без модели и любых новых проверок, включая CPU tests, Verify, Preflight
и GPU probes. Целевой прогон 15 октября отложен до Start; ниже он остаётся
будущей задачей после возобновления пользователем. Новая полная серия
в ближайшую очередь не входит. Эта актуализация не даёт команду на запуск,
commit, merge или push.

Открыты два разных вопроса: снижение скорости после soak и остановка
отдельного длинного запроса. Обратимый short gain 25–28% от CPU affinity
подтверждён, но не доказывает исправление длинного stall или причину прежнего
снижения. Память в доступных samples неудачного probe выше runtime-порогов;
в наблюдении есть разрыв 29,863 с. Причина этого разрыва тоже не установлена.

Подробности реализации: [progress diagnostics](backend-progress-diagnostics-20261013.md),
[штатный запуск](run-windows-rocm64k.md). План от 02.10 и датированные записи
в конце этого файла сохраняют историческую область результата.

## 1. Цель и точка начала

Интегрировать полезные исправления и выбранные обновления для **native Windows, RX 9070 XT / gfx1201, Qwen3.6-35B-A3B GGUF Q4_1, KV BF16, контекст 65536**, сохранив контрольный runtime и возможность отката каждого изменения. Основная интеграция выполнена; ближайший незавершённый этап — модельная приёмка новой progress-диагностики и получение evidence при возможном повторении stall.

**Исходная точка плана на 12.10.2026.** Таблица ниже описывает начало
интеграции; актуальные статусы и оставшаяся очередь приведены в разделе 2.

| Параметр | Зафиксированное состояние |
|---|---|
| Ветка / HEAD | `integration/research-refresh-20261002` / `e5e2899eef36e55eb84a58a4cd59de8766ac8c46` |
| Сохранённая база main/64K | `c4832f35055c5f27998fd36996d21d4dbda18b76`; Git-подготовка выполнена ранее |
| Контрольный stack | CPython 3.12.0, AMD Torch 2.13.0+rocm10.0.0, ROCm 10.0.0, Triton Windows 3.8.0.post28, TVM-FFI 0.1.13.post3 |
| Система | Windows11 build 26100 / 24H2, DriverStore 32.0.31041.1004 |
| Профиль | naive cache, KV BF16 / 65536, prefill 512, MoE 2048 / LRU, graphs 0, concurrency 1, serial загрузка экспертов |
| Уже реализовано локально | Opt-in first-NaN diagnostics, zero-padding 39 masked FLA loads, часть int64 offsets; изменения ещё в рабочем дереве |
| Приёмка начатого этапа | CPU/address и GPU operators подтверждены; новый diagnostic и ordinary short/63488/short прошли, оба clean Stop |
| Прежний отказ | Исторический ordinary long smoke дал NaN после4096prefill/forward253; после интеграции отказ не повторился, историческая первопричина отдельно не доказана |

Исторические успешные64K серии сохраняют свои результаты. Они не квалифицируют новые source/package combinations автоматически. Сохранённый opt-in диагностический механизм и текущие local changes продолжаем использовать, а не реализовывать заново.

Основания плана:

- [Отчёт начатого offline этапа](integration-stage1-offline-20261002.md).
- [Предыдущий план](research-integration-plan-20261002.md).
- Дополнительный локальный аудит: `rocm10-20260926/research-refresh-20261011-audit/` от корня общего workspace; `AUDIT_RU.md`, `candidate_decisions.json`, `bounded_package_candidates.json`, `coverage.json`.
- В аудите проверены82 имени пакетов, 53 refs в 45 репозиториях, статусы 81 PR / 54 issues и 19 старых PR; вручную сопоставлены выбранные diff. Наличие публикации или чужого benchmark не означает нашу квалификацию.

Срез версий и revisions относится к 11 октября UTC / 12 октября по Москве. Перед применением освежаем только выбранный источник и его artifact hashes.

## 2. Готовность и оставшаяся очередь на 15.10.2026

### 2.1. Что уже готово

| Этап / результат | Статус по сохранённой приёмке | Что остаётся в этой области |
|---|---|---|
| 0 — baseline/provenance | Manifests, donor revisions, окружения и отдельные caches сохранены | Новый снимок нужен при следующем изменении исполняемых inputs |
| 1 — FT593/599/527 | Conversion protection, KV accounting и sampler contracts приняты | Повторного переноса нет |
| 2 — FLA/GGUF/64K | Operator contracts, diagnostic и ordinary short/63488/short прошли, оба Stop чистые | Прежняя причина NaN отдельно не установлена; новый stall 13 октября учитывается отдельно |
| 3 — server/tokenizer/cache lifecycle | FT608/650 и live resize приняты; FT641 неприменим к локальным функциям | Повторная реализация не требуется |
| 4 — Python dependencies | Приняты выбранные группы, 74-package manifest и GGUF/API compatibility | TVM-FFI/Numba и обновления вне выбранных bounds — условный backlog |
| 5 — ROCm 10.1 / Torch 2.14 / post29 | Q4_1, attention, short/load, полная серия 353 requests и Stop прошли; рабочий config переключён | Compiler hook обязателен; upstream compiler не исправлен, общий speed gain SDK не доказан |
| 6.1 — portable Triton knob | Source opt-in реализован и функционально принят | Default off: нового выигрыша не показано |
| 6.2 — routed-only Q4_1 short prefill | Ограниченный opt-in принят; TTFT benefit подтверждён для короткого профиля | Default 0; overlap/long не квалифицированы, расширение опции отложено |
| 8 — capacity-срез 4K/8K/64K | На ROCm 10.0: 9/9 correct, все Stop чистые; 37,20/37,42/37,56 TPS | Это историческое сравнение capacity, не новый SDK benchmark |
| Штатный launcher 64K | Перенесён в repository, обновления SDK и affinity слиты/запушены в main | Существующую интеграцию повторять не нужно |
| Backend CPU affinity | Два обратимых A/B показали +25–28%; repository bootstrap и короткий smoke приняты | Локально выбран `one_per_core`; длинный prefill и soak этого режима не квалифицированы |
| Progress diagnostics | Stage journal, C watchdog, CaptureDiagnostics и capture перед Stop реализованы; 118 CPU checks прошли | Модельная приёмка отсутствует; `progress_diagnostics_model_qualified=false` |
| Сводка сохранённых логов | Завершена 15 октября; хронология, память, скорость, Stop и ссылки на raw evidence | Использовать как исходные данные следующего этапа |

Этап 7 — prefix reuse/FP32 intermediate state/workspace — остаётся
необязательным развитием. Он не включён в ближайшую очередь.

### 2.2. Что осталось и в каком порядке

**А. Принять новую диагностику с моделью — ближайший незавершённый этап.**
Сейчас отложен: пользователь запретил модель и новые проверки. После
явного возобновления выполнить одну целевую последовательность, readiness
учитывать отдельно:

1. Сохранить точные source/runtime inputs, включая локальные изменения,
   config, compiler workaround и CPU affinity. Одного HEAD недостаточно.
   До загрузки действуют startup RAM/commit/GPU free 38/45/14 GiB,
   runtime 12/4/3 и recovery 38/28/14. Прежние разовые исключения не переносить.
2. `Start -ProgressDiagnostics` с текущим `one_per_core`; подтвердить
   context/KV 65536, PID/time и включённый recorder.
3. Ровно три задания: короткий контроль 24 tokens → прежний
   `decode-63488` → тот же короткий контроль. Ожидаемый counting-ответ
   каждого — 391 token и сохранённый SHA256 `34a4f1e5…`.
4. Сохранить stage journal, Python stacks и capture metadata до Stop;
   записать request ID, PID/time и факт fresh ack. При отсутствующем ack
   не объявлять fresh stack capture успешным.
5. Выполнить managed Stop и сверить завершение записанных процессов,
   свободный порт и итог ресурсов. На первом fault закончить попытку,
   сохранить первый fault marker и доступные stacks/latest.

**Условие завершения А:** три корректных ответа, освобождённый KV после
запросов, работающий recorder с согласованной identity, успешный capture
и чистый Stop. При ошибке сохраняется незавершённый статус, следующий шаг
определяется полученным evidence. Diagnostic timing не квалифицирует скорость.
Полная серия, soak и повторные 4K/8K замеры сюда не входят.

**Б. Разобрать результат и закрыть только подтверждённую область.**
Если А успешен, записать модельную приёмку diagnostics для конкретных
source/runtime/config; обновить report, manifest и соответствующие release
flags. Это не доказывает причину или устранение исторического stall.
Если А воспроизводит отказ, сопоставить UID, последний stage, watchdog stack,
heartbeat, ресурсы и порядок cleanup; затем выбрать один узкий reproducer
по конкретной гипотезе. Native HIP/GPU stack новый recorder не предоставляет.

Открытые вопросы вести отдельно:

- Длинный probe: `UID=37`, prefill counter 54 272/63 488, нет terminal reply,
  managed Stop failed; позднейшее освобождение ресурсов не заменяет Stop.
- Наблюдатель: разрыв heartbeat 29,863 с и process probe timeout;
  источник задержки неизвестен.
- Производительность: после soak 50,02 → 39,06 TPS; эффект affinity
  подтверждён в другом запуске, размещение потоков в момент снижения неизвестно.

**В. Оформить изменения progress diagnostics.** После результата А/Б
обновить область qualification и подготовить отдельный commit. Commit,
merge и push — только по новому поручению пользователя; предыдущая публикация
launcher/SDK/affinity уже выполнена. Служебный `.serena/` и raw evidence
не включать в функциональный перенос. Сейчас изменения остаются локальными.

### 2.3. Отложенное развитие

Эти пункты не блокируют ближайшую приёмку diagnostics и не запускаются
автоматически после неё:

- Узкое исследование сохранения скорости с `one_per_core`, если пользователь
  вернётся к этому вопросу: нужны размещение потоков и обычные timings в момент
  снижения. Нынешние +25–28% не являются приростом от SDK.
- Замена compiler workaround подтверждённым исправлением компилятора.
  Текущий hook сохраняется; будущий candidate требует собственных compiler,
  operator и затронутых model gates.
- Обслуживание ОС/драйвера для другой строки vendor support — отдельная
  системная задача. Работа на сохранённой 24H2/старом драйвере не равна
  квалификации всей официальной матрицы; обновление хоста сейчас не выполняется.
- TVM-FFI/Numba, Attention632, bounded workspace, GEMV/GDN fusion,
  расширение routed-only demand на overlap/long и prefix reuse этапа 7 —
  только при подтверждённой пользе/активном callsite и отдельном объёме работы.

Параметры, exact donor revisions и способы отката для завершённых и
отложенных этапов сохранены в разделе 4. Датированные записи раздела 6
описывают состояние на дату записи и не заменяют актуальную очередь выше.

## 3. Общий порядок работы

1. Одна смысловая правка — отдельный reviewable diff. Записывать donor revision и attribution; переносить необходимые hunks. Commit/push выполняются при пользовательском поручении.
2. Существующие FLA/diagnostics изменения, новые переносы, dependency bounds и экспериментальные features учитывать отдельно. Служебный `.serena/`, raw dumps, wheels, модели, venv и kernel caches не включать в функциональный перенос.
3. На source A/B сохранять packages/driver/checkpoint/settings. На dependency A/B замораживать source и менять одну согласованную группу. Для нескольких факторов вывод относится ко всему выбранному комплекту.
4. Для hardware этапов соблюдать [ROCm baseline](rocm-experiment-baseline.md): Torch до Triton, проверенный imported source, отдельные native builds/caches, один сервер на GPU, graphs 0, explicit offload strategy.
5. Сохранённые startup 45 GiB free commit и runtime RAM/commit/local-VRAM 12/4/3 GiB, observer/watchdog и managed Stop действуют. Recovery/quarantine проходит штатную admission процедуру; успешный CPU test не снимает marker.
6. Diagnostic mode с finite checks/snapshots меняет timing. Скорость публиковать только с выключенной диагностикой и из обычного execution path.
7. Первый non-finite/device fault/watchdog/memory violation/неудачный Stop завершает попытку. Новый запуск оправдан исправлением или новой проверяемой гипотезой.
8. Для каждого этапа статус разделять на source/CPU, operator, model и performance. Не объявлять GPU часть завершённой по AST/mock/interpreter проверке.

### Минимальные проверки

| Изменение | Минимальный объём |
|---|---|
| Conversion/stats/API contracts | Один компактный набор CPU fixtures для реально изменённых случаев; запуск только затронутых tests |
| GDN/quant/index math | Существующий numerical fixture плюс обнаруженный новый edge case; независимый reference |
| Native/compiler изменение | Один маленький operator reproducer и metadata compilation/LDS, без полной модели |
| Локализация прежнего NaN | Один managed diagnostic launch: readiness, short, прежний long; остановка на первом отказе |
| Ordinary приёмка после исправления | Один short → long → short и clean Stop |
| Одна оптимизация | Один warmup и 3 operator timings на вариант либо до 6 коротких model A/B requests; использовать только нужный уровень |
| Итог4K/8K/64K | По 2 коротких запроса на capacity; long 64K повторять лишь при отсутствии результата того же runtime |

Это бюджеты первой проверки, а не сумма обязательных прогонов каждого патча. Совместимые изменения можно объединить в один hardware smoke после раздельных CPU gates; новую библиотеку или новый kernel нельзя квалифицировать результатом другого source/binary. Широкая серия и длительный soak в первую очередь не входят.

## 4. Этапы реализации

### Этап 0 — сохранить контроль и provenance

**Результат:** baseline manifest и таблица существующих/новых изменений.

- Зафиксировать HEAD/branch, hashes dirty source, checkpoint identity, interpreter/packages, active source path, runtime settings и cache roots.
- Переиспользовать собранные inventories; не скачивать модель или все 82 пакета заново.
- Разрешить полные SHAs только выбранных open PR; проверить их diff против текущего дерева.
- Для каждого пакета сохранять URL, distribution version, wheel tags и фактический SHA256 байтов перед установкой. Source SHA и publisher digest не заменяют фактическую проверку файла.

**Готово:** checkpoint изменений понятен; control environment/build/cache доступны для возврата.
**Откат:** source baseline и сохранённый manifest без сброса рабочего дерева.

### Этап 1 — небольшие полезные изменения без модели

**1.1. Защита conversion output, [FreeToken593](https://github.com/FlashML-org/FreeToken/pull/593).** Revision `1d5ce5459b6632adc60feb50b07cd4bbf73af1c6`, open head на срезе.

Локальная область: `python/freetoken/checkpoint/convert.py`. Отказывать до изменения файлов при существующем завершённом FTW output; определить понятное поведение для частичного/прерванного результата и явного намерения повторить conversion. Имена новых CLI опций определяются при реализации.

**Gate:** tiny file fixtures для отсутствующего, законченного и interrupted output; существующие bytes сохраняются при отказе. Полная модель и conversion большого checkpoint не нужны.

**1.2. Реальная статистика KV, [FreeToken599](https://github.com/FlashML-org/FreeToken/pull/599).** Merged `581fdca310d429cca74ac9d94ed2b893b86d9320`.

Локальная область: `python/freetoken/server/stats.py` и фактический backend stats contract. Capacity/usage вычислять из backend geometry, сохранить смысл полей shell/API. Различать configured context, effective capacity, occupied pages и actual request tokens.

**Gate:** fake backend с пустым cache, разной page geometry и изменённой capacity; проверка деления при нулевых значениях и accounting. Model launch не нужен.

**1.3. Штрафы за повторения, [FreeToken527](https://github.com/FlashML-org/FreeToken/pull/527).** Merged `c8186396f5834994628f7750b11da7300e1d4714`.

Локальная область: `core.py`, `engine/sample.py`, `server/api_models.py`, `server/generation.py` и существующий adapter. Провести presence/frequency параметры от API до logits, добавить request-owned counters и finite/range validation. Нулевые penalties сохраняют обычный путь; counters создаются только когда нужны и освобождаются с request. Сохранить greedy/mixed batch, token accounting, cancellation/Stop.

**CPU gate:** диапазон [-2,2], NaN/Inf rejection, параметрная цепочка, подсчёт повторов и reset на новом request, mixed greedy и penalties 0. GPU operator gate — один маленький logits fixture позднее; его можно совместить с hardware окном 2. Не заявлять устранение прежнего prefill NaN изменением sampler.

**Откат этапа 1:** отдельный source diff каждого пункта. Полезные conversion/stats исправления не зависят от результата long diagnostic.

### Этап 2 — завершить текущую 64K корректность

**2.1. Сопоставить остаток [FLA1329](https://github.com/fla-org/flash-linear-attention/pull/1329)**, merged `b4a0c7344078f28cd5acfee9533feba8c9213dae`.

- Сохранить существующие masks/padding/int64 fixes.
- Проследить remaining recurrent decode indices: `kernel/fla/fused_sigmoid_gating_recurrent.py`, state slot/program IDs и multiplication chains; расширять до int64 до арифметики только там, где размерный контракт этого требует.
- Проверить flattened grid assumptions. Не менять GQA, state[V,K], dtype/precision и in-place lifecycle по аналогии с большой upstream реализацией.
- Переиспользовать `tests/kernels/test_fla_padding.py`, добавив только непокрытый адресный случай>=2^31 без большого allocation. Связь overflow с прежним NaN не доказана.

**2.2. Проверить active GGUF launch contract, [FreeToken185](https://github.com/FlashML-org/FreeToken/pull/185)**, open head `3ca0ccd0365cfe803c959221b7ea442e7167c8f2`.

Локальная область: `kernel/csrc/gguf/moe_vec.cuh`, `moe/fused_q4_0.py`. Текущие19launch sites используют grid.z=tokens*top_k. Проверить реальные HIP maxGridSize и error-check semantics, затем изменить flattening/launch guards при подтверждении ограничения. Сохранить плотный expert-major Q4_1 stride и affine quant math; пройти both gate/up и down mapping.

**Gate:** tiny dimensions и реальные boundary counts при topk 8, CPU mapping/reference и один native operator fixture; новый native build/cache. Не увеличивать chunk 512 ради проверки. 64K capacity не равна kernel batch64K, и текущий512*8=4096 сам по себе не достигает donor CUDA предела65535.

**2.3. Проверить только пробелы 134/638.** HIP optional-backend probes, TensorMatcher, zero bounds/options и PDL fallback частично уже локальные. Если конкретного отсутствующего guard нет, завершить пункт отметкой об эквивалентности.

**2.4. Завершить начатую диагностику.** Проверить наследование opt-in environment в managed backend, imported source, актуальный cache/build и compiler LDS. Один short+long с существующим ограничением first-NaN snapshots 64 MiB. При первом нарушении построить маленький reproducer и исправить найденную операцию. Если отказ исчез, отдельно выполнить ordinary smoke с диагностикой off.

**Готово:** затронутые math/address/launch contracts проходят; exact candidate runtime завершает прежний long и clean Stop. Успех может подтвердить устранение воспроизводимого отказа, даже если историческая первопричина не доказана; это записывается отдельно.
**Откат:** control source/native cache; сохранить first-failure evidence. Время диагностики не считать benchmark.

### Этап 3 — API, tokenizer и cache lifecycle

Каждый пункт имеет собственный CPU gate; изменения engine/cache требуют ordinary correctness из 2 перед принятием.

**Выполнено 12.10.2026:** API/tokenizer608 и cache lifecycle650 адаптированы. 243 CPU tests прошли, DSV4 physical footprint дополнительно проверен; три коротких model requests и один mixed-direction live resize прошли, Stop чистый. 641 неприменим к нашей версии: обе donor Torch fallback-функции отсутствуют. Подробности в [отчёте этапа3](integration-research-stage3-20261012.md).

| Кандидат | Что переносить | Gate |
|---|---|---|
| [FreeToken608](https://github.com/FlashML-org/FreeToken/pull/608), `93b0a59164bb6d68749298f868ca3738fd464143` | Поэтапно shared tokenizer/message handling, затем OpenAI/Anthropic/CLI adapters | Tool turns и порядок сообщений, inline positions, streaming/non-streaming, token budgets/cancellation; около 55k-character diff не переносить одним необследованным блоком |
| [FreeToken641](https://github.com/FlashML-org/FreeToken/pull/641), `3cae53d415518a53faabac4f099d6be230f63e91` | FP32 mRoPE alias fix после tiny воспроизведения в нашем wrapper | FP32 и BF16, input immutability, independent rotary reference; BF16 .float() copy не считать FP32 alias |
| [FreeToken650](https://github.com/FlashML-org/FreeToken/pull/650), `4f4832839719cb814f928821cc40de3336fd4592` | Shrink-before-grow для наших real cache/pools, затем rollback/reference consistency | Mock free/allocate ordering и allocation failure; один bounded live resize только если используется этот путь; donor DSV4-specific файлы могут отсутствовать у нас |

**Готово:** включённые contracts подтверждены. Пункт, который уже покрыт или не относится к нашим paths, получает итоговое решение вместо обязательной реализации.
**Откат:** вернуть один source diff/adapter или старый cache rebuild contract, сохранив abort/Stop.

### Этап 4 — выбрать небольшие dependency groups

**Выполнено 12.10.2026:** tokenizer и API groups приняты в отдельной среде на прежнем native AMD stack. Source adapter сохраняет точный Qwen tokenizer contract; ModelScope optional. 259+3 CPU tests и три коротких model requests прошли. TVM-FFI и Numba/llvmlite обновления отложены по собственным gates. [Отчёт этапа4](integration-research-stage4-20261012.md).

Версии ниже — проверенные кандидаты среза, не команда обновить всё.

| Группа | Candidate | Условие |
|---|---|---|
| Tokenizer/model tooling | Transformers 5.19.0 + Hugging Face Hub 1.33.0 | Transformers требует Hub>=1.31; 1.33 укладывается в project<2. Проверить embedded GGUF metadata/converter, vocab/AddedTokens/stop IDs/chat template, combining marks и token-budget accounting |
| API | FastAPI 0.143.0 + Pydantic 2.14.1 / core 2.50.1 при подтверждённой пользе | Полный resolver manifest, новые transitive dependencies, HTTP/streaming/schema compatibility. Core жёстко связан с Pydantic |
| Native FFI | TVM-FFI 0.1.14.post1 | Source hard pin меняется отдельным diff; native ABI/JIT/build gate относится к 5 |
| CPU JIT | Numba/llvmlite согласованной парой | Сначала подтвердить active use и joint dependency constraints; при отсутствии пользы отложить |
| Packaging | ModelScope | Устранить source-vs-installed discrepancy для поддерживаемого entry point: необходимая install group либо optional dependency с guarded imports; для чтения имеющегося GGUF не устанавливать только ради списка |

Сохранить OpenAI 2.54.0, NumPy 2.4.6 и flashlib 0.3.0. OpenAI 3.x, Hub 2.x и NumPy 2.5.x выходят за действующие bounds и требуют самостоятельной migration. PyPI rocm* placeholders и upstream Torch wheel не подменяют выбранные AMD distributions.

Token-only checks выполнять на metadata fixtures/существующем tokenizer loader. Современный Transformers GGUF model load может использовать другую packed/MPS ветку или dequantization fallback; это не замена нашего native HIP engine. [Официальные GGUF docs 5.19](https://huggingface.co/docs/transformers/v5.19.0/quantization/gguf).

**Готово:** constraints и installed Requires-Dist согласованы, нужные entry points/token IDs сохраняются; hardware smoke добавляется только при изменении active inference path.
**Откат:** предыдущий manifest в control среде; не наслаивать следующие группы на failed candidate.

### Этап 5 — отдельный vendor stack и compiler qualification

**Актуальный результат на 15.10.2026:** ROCm 10.1 / Torch 2.14 / Triton
post29 функционально принят с обязательным compiler workaround и выбран
в штатном 64K config. Q4_1, attention, short/load и полная серия 353 requests
с clean Stop выполнены. Изменения слиты и отправлены в main. Общий speed gain
SDK не подтверждён; native compiler не исправлен. Отдельный длинный probe
позднее остановился, его причина остаётся задачей раздела 2.
[Полная серия](rocm1010-full-series-20261013.md),
[обновление библиотек](rocm1010-upgrade-20261012.md),
[сводка](C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/64k-saved-logs-summary-20261015.md).

**История первого gate, 12.10.2026:** stable 10.1.0 заменил RC-план.
Девять vendor packages обновлены в отдельной CP312 venv; 74 installed packages,
native build/load и CPU TVM interop прошли. Обе Torch bounds — `>=2.11,<2.15`.
Исходный post29 ошибочно сворачивал histogram и два loop comparisons, поэтому
первая runtime acceptance не прошла. Позднее явный repository compiler hook
обошёл ошибочный pass и прошёл выбранные compiler/operator/model gates;
vendor files не менялись. Первая неудачная квалификация сохранена в
[историческом отчёте этапа 5](integration-research-stage5-20261012.md),
[exact artifacts](../constraints/windows-rocm1010-artifacts-20261012.json) и
[constraints](../constraints/windows-rocm1010-candidate-20261012.txt).

| Вариант | Состав | Назначение |
|---|---|---|
| Rollback/control | AMD Torch 2.13.0+rocm10.0.0, SDK 10.0.0, Triton post28 | Сохранённая отдельная рабочая среда |
| Принятый рабочий stack | AMD Torch 2.14.0+rocm10.1.0, matching device-gfx12-0 и SDKcore/devel/libraries/device-gfx1201=10.1.0; Triton Windows 3.8.0.post29 | Штатный 64K, отдельные venv/native build/cache; compiler hook обязателен |
| Отложенная diagnostic развилка | AMD Torch 2.13.0+rocm10.1.0 с matching components | В исходном плане — различение SDK/Torch эффекта. Не реализована, не входит в оставшийся обязательный объём |

Порядок квалификации артефактов, выполненный для принятого комплекта и
сохраняемый для будущей замены compiler/runtime:

1. Получить exact artifacts, SHA256 байтов, metadata и resolved manifest; `rocm==10.1.0` metapackage является sdist, и его build metadata требует отдельного учёта.
2. Точечно согласовать **runtime и build-system** Torch bounds: выполнено, оба `>=2.11,<2.15`. Не открывать верхнюю границу произвольно для будущих major releases.
3. Проверить CP312/win_amd64 tags, Torch internal APIs и native extension ABI, SDK/DLL provenance и импорт нужного checkout.
4. Проверить выбранные маленькие compiler reproducers и LDS до model smoke. Triton 11577 histogram backport был open, 11429 loop backport closed unmerged; post29 source не содержит подтверждённого histogram fix. Источник fixes выбирается по failing operator, не по обещанию версии.
5. Отдельный managed ordinary smoke только после static/native gates; сохранить exact source/package/compiler/DLL manifest.

**Сохранённый срез официальных источников на 12.10.2026:**
[индекс AMD](https://stable.repo.amd.com/rocm/whl-next/) и
[официальная установка](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
содержали Torch 2.14 / gfx1201 пару;
[матрица 10.1](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
указывала Windows 11 / 25H2,
[driver release](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html) —
DriverStore 32.0.31041.5005. Записанная машина 24H2 / 32.0.31041.1004
отличается. Срез 15 октября не обновлялся; результат на этой машине имеет
указанную область, а не подтверждение всей vendor матрицы. Venv не изолирует
системный driver: обслуживание ОС/драйвера — отдельная задача с новой
фиксацией control после изменения хоста.

**Готово в принятой области:** manifest/native/compiler workaround/operator
и функциональная модельная приёмка записаны; обычный запуск переведён на
этот stack. Performance qualification остаётся открытой; снятие compiler
hook и приёмка другой ОС/драйвера требуют отдельного evidence. Повторять
уже выполненные Q4_1/attention/full-series gates для обновления этого плана
не требуется.
**Откат:** control venv, native binaries и cache roots. После системного обновления возвращение venv не возвращает старый driver; контрольные замеры надо актуализировать.

### Этап 6 — portable source optimization и одна измеренная оптимизация

**6.1. Сначала ранее подтверждённый process-start Triton knob.** Лабораторные helpers устраняли повторное чтение getter и дали исторический gain. Перенести минимальный инициализатор в repository source как opt-in для проверенного AMD runtime: после Torch/SDK и до первых kernels закрепить публичным assignment **то же эффективное bool** `triton.knobs.amd.use_buffer_ops`. Сохранить explicit overrides; unsupported descriptor/version пропускать. Не редактировать установленный Triton.

**Gate:** true/false, explicit override, unsupported runtime, once-per-process и fresh process rollback; короткий source smoke и малый A/B без profiler/trace. Обязательный logits safety every64 для capacity>8K сохраняется одинаково во всех phases. Старые +27.44% не записывать как новый результат.

**6.1 выполнен 12.10.2026; ускорение не подтверждено.** Source opt-in добавлен, default off. CPU28passed; main/off/on — по одному warmup и двум measured requests, все9ответов одинаковы, все Stop чистые. Медианы52,32/51,82/51,73TPS; ON/OFF−0,16%, меньше observed spread. Functional gate пройден, оснований включать флаг по умолчанию нет. Использован прежний qualified AMD10.0/Torch2.13/Triton post28; failed stage5 stack не применялся. [Отчёт этапа6.1](integration-research-stage6-20261012.md).

**6.2. Затем выбрать один фактический bottleneck.**

**6.2 выполнен 12.10.2026 для opt-in короткого Q4_1 prefill.** Перенос всех256экспертов слоя при24token rows заменяется admission выбранных experts через cache. Порог0–32, default0; только ROCm/native Q4_1/GPU decode без overlap, raw routes ограничены256 и capacity. CPU42passed/5skipped; два отдельных native GPU contracts прошли побитное сравнение и eviction/hit/reference checks. Modelmain/off/on9/9 и cleanStop: TTFT0,8205/0,8280/0,2500s, TPS40,52/40,33/39,53. Новый TTFT benefit−69,81% кOFF подтверждён в этом малом workload; decode gain не установлен. Ordinary overlap profile не переключён. [Отчёт этапа6.2](integration-research-stage6b-20261012.md).

| Кандидат | Почему рассматривать | Решение до запуска |
|---|---|---|
| [Attention632](https://github.com/FlashML-org/FreeToken/pull/632), `078132480ea4a0424f677a27eb22085624d331e6` | Наш active full-attention prefill | Independent numerical fixture + gfx1201 LDS; затем короткий A/B |
| Routed-only short prefill, `ff4966bd3b57de03f1163967d6c07de2a41d7ee9` | Сократить загрузку неиспользуемых экспертов | Выполнено для ограниченного Q4_1 profile: remap/eviction/duplicates прошли; default0, TTFT benefit в отчёте6.2 |
| Bounded MoE workspace, `196eace47a9df5937113274a61fa459a594bb8fd` | Peak-memory contract | При chunk 512 показать оставшийся peak; если пользы нет, не добавлять sub-launches |
| Dense/router GEMV, `fdf015aeed5f30b1c89b9fea51e41ff2bc5993a0` | Single-row dense cost | Проследить GGUF dense dispatch; Q4_1 и F32 GDN slots не подменять BF16 donor; near-tie router topk/reference |
| GDN view/Conv1D fusion | Остаточные materialization/launches | Не переносить уже существующую fusion; CDNA geometry требует собственной qualification |

Источники и остальные точные revisions сохранены в этапе 5 [старого плана](research-integration-plan-20261002.md). Shared activation quant/router patches, RDNA native vector math и sync removal выбирать только после отдельного layout/lifetime/byte-parity evidence. Отозванные donor 2–4x NaN-routing числа исключены из целей.

**Готово:** effect на нашем path больше observed measurement spread; TTFT/decode/wall/memory записаны, correctness сохраняется. При шуме — результат «не определён», при регрессии — откат/default-off.
**Откат:** flag off и fresh process или один diff; отдельные compiler/cache namespaces.

### Этап 7 — optional prefix reuse, отдельная функция

Это отложенное развитие; оно не требуется для выпуска исправлений 1–6.

Порядок из предыдущего плана: единый workspace reserve для startup/rebuild → FP32 intermediate h с учётом удвоения BF16 workspace → final-short-chunk snapshot lifecycle → bounded pinned host bank/reclaim. Использовать точные donors старого плана, не менять naive profile на radix автоматически.

Новые идеи jomcgi/Windows/ox-boost forks относятся к bounded governor, VirtualLock/pin budgets, tier restore ownership и completeness snapshot writer. Их NVIDIA/Linux/KDA/NVFP4 runtime не переносится целиком. Сначала проектировать соответствующий HIP/GDN contract.

**Gate:** маленький cold-vs-resume state/logits fixture, chunk/partial boundaries, eviction/rebuild, ownership и отсутствие pinned growth; затем один long prefix+continuation в отдельном opt-in radix profile. Host slots default 0; host bound рассчитывается по свободному reserve.
**Откат:** naive/default 0 и fresh process; planner/precision/lifecycle changes остаются отдельными.

### Этап 8 — итоговый срез и принятие

**Завершён12.10.2026:** один runtimePython74/native10.0/post28, ordinaryoverlap/labcachedknob; sourceknoboff/demand0. Warmup+2measured на capacity: 4K37.20 / 8K37.42 / 64K37.56TPS, 9/9correct, cleanStop. PrefillTPS отдельно не измерен. Полный final-tree long сегодня не повторялся; preservedstage2long и numerical/native inheritance имеют явную область. [Отчёт](integration-research-stage8-20261012.md).

- Зафиксировать один exact manifest и исходники принятых этапов; записать отклонённые/невыгодные кандидаты с основанием.
- По 2 запроса одинаковой короткой задачи при configured capacity 4K/8K/64K, одинаковые generation budget/cache/penalties/knob/driver/Moonlight условия. Записать actual prompt tokens отдельно от capacity.
- Это сравнение расходов разных capacity при одинаковом workload. Скорость после фактического длинного prefix — другая строка; не смешивать её с короткими запросами.
- Long64K+Stop не повторять, если уже есть обычный результат того же source/binary/settings и затронутые изменения его не обесценивают.
- Таблица: correctness, TTFT, prefill throughput, decodeTPS, totalwall, generated count, RAM/commit/VRAM minima, source/package/compiler hashes и размер выборки. Диагностические timings выделять отдельно.
- Исторические short medians 4K 40.05 / 8K 40.27 / 64K 41.40 TPS сохранить как reference; итоговый 64K сравнивать с новыми 4K/8K той же серии. Отсутствующий замер обозначать как отсутствующий.

**Готово:** полезные принятые changes воспроизводимы, обычный 64K проходит, результаты/откаты документированы. В этой итерации готовность не зависит от реализации optional этапа 7 или каждого условного кандидата. Публикация/merge выполняется пользовательским поручением; текущий план её не подтверждает.

## 5. Полный учёт новых кандидатов

| Кандидат | Решение |
|---|---|
| FT527,593,599 | Выполнено в этапе 1; повторный перенос не нужен |
| FLA1329, FT185 | Выбранные локальные contracts интегрированы и приняты в этапе 2 |
| FT134,638 | Этап 2.3 завершён: недостающий guard134 перенесён, остаток638 покрыт локально |
| FT608,641,650 | Этап 3 завершён: 608/650 приняты, 641 неприменим |
| FT632 | Условный backlog этапа 6: active-path audit, numerical/LDS и performance gates при выборе кандидата |
| FT138 | Дубликат guards; повторный перенос не нужен |
| FT56 | Условный backlog: empty host_register callsite/ownership contract. Clamp nbytes<=0 к 1 byte не считать безопасным для null/negative |
| FT495 | Условный backlog: ZMQ subscription barrier только после локального CPU startup reproducer |
| FLA1247 | Локальный fixed-config путь уже эквивалентен; повторного переноса нет |
| FLA1359 | Отложить batch/JIT specialization при нынешнем B=1 |
| AITER5624 | Canonical input_guard уже делает contiguous; direct l2norm callers аудировать при необходимости |
| AITER6321 | KDA-specific, нынешний GDN is_kda=False |
| AITER4306 | Отдельные Windows JIT build techniques; библиотека не готовый универсальный accelerator |
| AITER5732 | FlyDSL/CDNA GDN, исследовательский backlog |
| SGLang43570 | CUDA skewed mixed-batch; concurrency 1/chunk 512 не его workload |
| Triton 11577,11429 | Источники compiler-гипотез среза 12.10; локальный обход принят, upstream fix не подтверждён. Не снимать hook по статусу PR |
| ROCm hipBLASLt11340 | Проверить включение в выбранный binary и active caller при SDK гипотезе |
| ROCm SDMA issue12535 | Данные gfx1100 hang, не доказанная причина нашего gfx1201 freeze |
| FT issue648 | Этап 4 принят для нашего GGUF contract; повторно учитывать при следующем изменении tokenizer |

Полные source revisions и первичные URLs этих 26 строк сохранены в audit ledger. Для merged переносов используется merged commit, для open — head SHA; непустой merge_commit_sha без merged_at не означает merge.

FreeToken-Kai, freetoken-rdna3, ZLUDA, nix-freetoken, full AITER/FlyDSL, сторонний engine, quantizedKV, MTP/speculative и CPU/DISK tier не входят в ближайшую интеграцию: требуются свои platform/layout/quality/lifecycle evidence. `radeon_ops` с нужным API остаётся неподтверждённым; одноимённый ComfyUI package не является заменой.

## 6. Журнал выполнения

Таблица актуализирована на 15.10.2026 по сохранённым отчётам. Датированные
записи ниже — история последовательных состояний; проверки в этой
актуализации не запускались.

| Пункт | Статус сейчас | Следующее действие |
|---|---|---|
| Git main/64K база и integration ветка | Launcher, SDK и CPU affinity слиты/запушены; база `dac3d655` | Локальную diagnostics работу продолжать в `integration/backend-progress-diagnostics` |
| Дополнительный аудит | Выполнен | Освежить только выбранный revision перед переносом |
| Этот план | Актуализирован 15 октября по сводке логов | Ближайшая незавершённая задача — раздел 2.2.А; сейчас модель и проверки отложены |
| Baseline/provenance этапа0 | Source manifests, donor metadata, GPU identity, отдельные caches и model/operator evidence сохранены | Для следующих source/package changes создавать новый manifest |
| FLA diagnostics/padding/int64 | CPU/address, GPU и model gate завершены для текущего профиля | Сохранять доказательства; не повторять широкую серию |
| Этап1.1 / FTW593 | Source/CPU завершены: 5 passed | Использовать отказ для готового/прерванного output |
| Этап1.2 / KV599 | Source/CPU завершены: accounting набор 14 passed | Backend page size + config fallback |
| Этап1.3 / penalties527 | Source/CPU и GPU sampler завершены | Request-owned counts и stream ordering подтверждены |
| Этап2 | Завершён: 4 GPU operator contracts, diagnostic3/3 + ordinary3/3, оба cleanStop | Данные и откаты в отчёте этапа2 |
| Этап3 | Завершён для текущего профиля: CPU243passed, tokenizer и live resize/API3/3, cleanStop;641 неприменим | Артефакты и границы в отчёте этапа3 |
| Этап4 | Завершён: отдельный74-package candidate, tokenizer/API parity, CPU259+3 и model3/3, cleanStop | Контроль сохранён; отчёт и constraints записаны |
| Этап5 | ROCm 10.1 с workaround принят: Q4_1/attention, short/load, full353 и clean Stop; рабочий config переключён | Сохранять hook и control10.0; native compiler fix/performance остаются отдельными вопросами |
| Этап6.1 | Portable knob реализован; CPU28passed, model9/9, cleanStop; новый gain не подтверждён | Сохранить default off; не повторять серию ради шума |
| Этап6.2 | Opt-in короткий Q4_1 prefill принят: CPU42 + GPU2 + model9/9, cleanStop; TTFT−69,81% | Default0; обычный overlap profile и длительная эксплуатация новой опции не квалифицированы |
| Этап7 | Отложенная optional функция | Не блокирует выпуск текущей итерации |
| Этап8 | Завершён: matched4K/8K/64K9/9, всеStopчистые; decode37.20/37.42/37.56TPS | Exact profile/ledger и ограничения в [отчёте этапа8](integration-research-stage8-20261012.md) |
| Обычный запуск после8 | Repository launcher подключён и опубликован; сейчас рабочий config64K/ROCm10.1/one_per_core | Не повторять перенос; [руководство](run-windows-rocm64k.md) |
| Полная серия ROCm10.1 | 353/353 correct, длинные входы, boundary, queue/cancel и clean Stop | После soak−21,91%: performance qualification открыта, новая полная серия не назначена |
| Backend CPU affinity | Обратимый A/B+25–28%; bootstrap и короткий smoke приняты, слиты/запушены | Long/soak режима ещё не приняты; short gain не считать исправлением stall |
| Progress diagnostics | Реализация и 118 CPU checks готовы; модель не запускалась, изменения локальные | Один целевой diagnostic short/long/short + Stop после возобновления пользователем |
| Сводка сохранённых логов | Готова 15 октября, причины stall/heartbeat gap не установлены | Использовать ссылки на raw evidence и условия завершения раздела 2.2 |

После каждого законченного пункта добавлять запись: donor/source revision, scope/diff, выполненная проверка и её уровень, результат, artifact path и способ отката. Невыполненные проверки остаются явно открытыми.

**Запись 12.10.2026:** перенесены контракты FT593 (`1d5ce5459b6632adc60feb50b07cd4bbf73af1c6`), FT599 (`581fdca310d429cca74ac9d94ed2b893b86d9320`) и FT527 (`c8186396f5834994628f7750b11da7300e1d4714`). Совместная CPU приёмка: **181 passed / 1 GPU skipped / 2 warnings, 7.57 s**; GPU не инициализирован. Раздельные patches, control source и logs сохранены в `rocm10-20260926/research-refresh-20261011-audit/stage1-20261012/` от общего workspace. Проверки и откат описаны в [отчёте](integration-research-stage1-20261012.md). Изменения остаются незакоммиченными; модель не запускалась.

**Запись 12.10.2026, этап2:** FLA1329 (`b4a0c7344078f28cd5acfee9533feba8c9213dae`), GGUF185 (`3ca0ccd0365cfe803c959221b7ea442e7167c8f2`) и отсутствовавший guard134 интегрированы; остаток638 уже покрыт локально. CPU FLA15passed/2skipped + backend2passed; 4 уникальных GPU contracts прошли. Diagnostic и ordinary short/63488/short — оба3/3pass; long391completion, ordinary total102.656s/TTFT95.172s. Content hashes совпали, KV released, оба Stop чистые. Старые quarantine/evidence сохранены. Подробности и артефакты в [отчёте этапа2](integration-research-stage2-20261012.md).

**Запись 12.10.2026, этап3:** FT608 (`93b0a59164bb6d68749298f868ca3738fd464143`) и FT650 (`4f4832839719cb814f928821cc40de3336fd4592`) адаптированы; FT641 (`3cae53d415518a53faabac4f099d6be230f63e91`) неприменим к отсутствующим локально fallback-функциям. CPU243passed, real GGUF tokenizer fold/count/prefix286, live resize KV65536→49152 / MoE2048→2112 и API3/3 прошли. Minima RAM27.115/commit14.284/VRAM5.501GiB, Stop чистый. Пакеты и профиль следующего запуска64K прежние. Подробности в [отчёте этапа3](integration-research-stage3-20261012.md).

**Запись 12.10.2026, этап4:** отдельный candidate с Transformers5.19/Hub1.33/FastAPI0.143/Pydantic2.14.1/core2.50.1 и OTel API1.44 проверен. Native AMD stack10.0 прежний; tokenizer/API contracts совпали с control после source compatibility adaptation, ModelScope optional. CPU259+3passed, model3/3passed, чистый Stop. Constraints и артефакты в [отчёте этапа4](integration-research-stage4-20261012.md).

**Запись 12.10.2026, этап5:** отдельный AMD10.1/Torch2.14/Triton post29 candidate подготовлен; exact artifacts, metadata/provenance и native build/load проверены, CPU298passed. Compiler неправильно сворачивает histogram и два loop comparisons, поэтому runtime acceptance не пройдена и model не запускался. Остаёмся на control10.0/post28. Подробности в [отчёте этапа5](integration-research-stage5-20261012.md).

**Продолжение 12.10.2026, обновление библиотек:** в ветке `integration/rocm1010-upgrade`
явный штатный Triton compiler hook пропускает ошибочный range-fold pass, не меняя
установленные файлы и отделяя compilation cache. CPU12passed, четыре compile-only
reproducers до/после, пять GPU reference checks и шесть project GPU tests прошли.
GPU-диагностика на прежних ОС/драйвере была отдельно разрешена пользователем;
model/long64K не запускались, обычный profile не переключён. Остались Q4_1/attention,
model/Stop, решение о host support и полезный результат. Подробности в
[новом отчёте](rocm1010-upgrade-20261012.md).

**Продолжение обновления, Q4_1/attention и short64K:** GPU3passed,
bootstrap CPU3passed, smoke client CPU4passed. Отдельный runtime с
Torch2.14/SDK10.1/post29 и compiler workaround выполнил readiness и один
request24/391tokens при live context/KV65536; reference hash совпал,
decode36.731TPS/TTFT1.786651s, Stop чистый. Minima RAM23.544/commit8.909/VRAM6.638GiB.
Первый model startup завершился LNK1104 из-за DLL path289characters;
CPU linker control289fail/119pass подтвердил причину, короткий state/cache
путь исправил повтор. История сбоя сохранена. Разовое пользовательское
разрешение на меньший commit применено только к пробному старту (42GiB),
после Stop возвращено45GiB. Полной серии/long64K не было; обычный profile
10.0/post28 прежний, изолированное ускорение не подтверждено. Подробности и
hashes в [отчёте](rocm1010-upgrade-20261012.md) и qualification manifest schema2.

**Применение13.10.2026:** штатный общий64K controller поддерживает pinned10.0/10.1,
default working config переведён на10.1 по запросу пользователя. Compiler
workaround обязателен, bootstrap до GPU init сохранён. CPU33+9passed,
Configure/Verify обоих окружений прошли. Actual short24/391 и load16384/292
валидны, KV released, Stop чистый; decode40.783/34.493TPS, load TTFT23.692341s.
Minima RAM24.918/commit13.577/VRAM6.620GiB. Две ошибки harness (CRLF и неверный
эталон391 вместо292) разобраны без изменения исходных failed reports;
post-load short не выполнялся. Новая full series/long64K отсутствуют,
performance gain не заявлен. [Profile](../constraints/windows-rocm1010-profile-20261013.json),
[запуск](run-windows-rocm64k.md), [отчёт](rocm1010-upgrade-20261012.md).

**Запись 12.10.2026, этап6.1:** public assignment сохраняет effective bool AMD buffer ops при opt-in и первом kernel import, unsupported runtime/descriptor пропускается, installed Triton не меняется. CPU28passed; main/off/on9успешных requests, identical292-token outputs, cleanStop. ON/OFF−0,16%; default off, новый speed gain не заявлен. Minima RAM25,261/commit13,127/local VRAM6,888GiB. Setup failures и восстановление стенда сохранены отдельно от результатов. [Отчёт этапа6.1](integration-research-stage6-20261012.md).

**Запись 12.10.2026, этап6.2:** donor `ff4966bd3b57de03f1163967d6c07de2a41d7ee9` адаптирован к native Q4_1 и проверенному bounded admission. CPU42passed/5skipped; два GPU contracts прошли, maxLRU LDS1024bytes. При configured64K и actual24/391tokens main/off/on дают TTFT0,8205/0,8280/0,2500s; decode40,52/40,33/39,53TPS. Все9outputs одинаковы, всеStopчистые. ON/OFF TTFT−69,81%; decode gain не принят. Пакеты/control сохранены, feature default0. [Отчёт этапа6.2](integration-research-stage6b-20261012.md).

**Исторический итог итерации 12.10.2026:** этап8 завершён; приняты source changes1–4 и opt-in возможности6, selected runtime — qualified Python74 с native10.0/post28. Новые опции6по умолчанию выключены. [Profile manifest](../constraints/windows-rocm1000-profile-20261012.json) и [итоговый отчёт](integration-research-stage8-20261012.md) фиксируют9/9matchedshort, прежнийlongscope и remaining gates. Optional7 отложен; stage5 compiler qualification и расширение demand на overlap/long на эту дату оставались отдельными задачами. Legacy release marker/quarantine не изменены; эта итерация не выполняла merge/publish. Более позднее применение ROCm10.1 и публикация отражены выше и в записи 13 октября.


**Запись12.10.2026, этап8:** одно и то же задание24/391tokens; decode37.20/37.42/37.56TPS при context/KV4096/8192/65536. Selected74packages, old native preserved, sourceoptionsdefaultoff, all9outputsidentical, all3Stopclean. 64K/4K+0.97%, 64K/8K+0.37%; это малыйcapacityсрез, не isolatedintegrationgain. Scope/reproducibility/release-marker границы в [отчёте этапа8](integration-research-stage8-20261012.md).

**Обычный запуск12.10.2026:** итоговое ревью завершено; source476files сохранён. Default64K/qualifiedPython74 подключён через working-release schema2, gate18 + sourceCPU9 прошли, один application request24/391 и cleanStop подтверждены. Штатный readiness request учитывается отдельно. История quarantine сохранена; Git commit/merge подготовлены. [Отчёт и запуск](integration-release-prep-20261012.md).

**Полная серия13.10.2026:** upgrade commit `f755ec7c` слит в `main` и запушен
в пользовательский fork. New native10.1/post29 с обязательным compiler hook
проверен из штатного launcher:353requests, exact64K boundary, repeated long
decode,6queue/cancel pairs,1837,5s/11cycles load,120s idle и cleanStop.
CPU61passed перед слиянием. Корректность/resources/lifecycle прошли;
matched control после soak50,02→39,06TPS (−21,91%), performance qualification
открыта, причина снижения пока неизвестна. Private memory−0,34497GiB
пересчитана из raw per-process samples; raw null aggregate не использован
как proof. [Отчёт](rocm1010-full-series-20261013.md),
[manifest](../constraints/windows-rocm1010-full-series-20261013.json).

**Исследование и CPU affinity, 13.10.2026:** сохранены 84 корректных коротких
запроса сравнения: old SDK 37,196 / new 36,175 TPS; ускорение SDK не доказано.
Два обратимых переключения backend на один LP физического ядра дали
+28,09%/+25,23%. Repository bootstrap принят отдельным коротким запуском:
1 readiness, 1 counting warmup + 3 measurements, median 45,882 TPS, чистый Stop.
Изменение слито и отправлено в `fork/main` как `dac3d655`.
Long/soak с affinity не квалифицированы; первый длинный probe с UID 37,
prefill counter 54 272 и неудачным Stop сохранён, его причина неизвестна.
[Исследование](rocm1010-speed-investigation-20261013.md),
[внедрение](rocm1010-cpu-affinity-launcher-20261013.md).

**Progress diagnostics, 13.10.2026:** в ветке
`integration/backend-progress-diagnostics` поверх `dac3d655` подготовлены
stage journal, C watchdog, bounded capture с PID/time/request ID и snapshot
до Stop. Сохранены 118 passed, включая 22 новых CPU cases. Модель/GPU compute
для этого механизма не запускались; qualification остаётся offline-only,
новые model flags false, изменения не закоммичены.
[Отчёт](backend-progress-diagnostics-20261013.md).

**Актуализация 15.10.2026:** целевой прогон с новой диагностикой отложен до
Start по просьбе пользователя. По сохранённым логам создана
[сводка](C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/64k-saved-logs-summary-20261015.md):
стадии исполнения, память, скорость и Stop имеют отдельные выводы. План
обновлён по этой сводке: завершённые переносы не возвращены в очередь,
остались модельная приёмка diagnostics, разбор её результата и оформление
изменений; исследования compiler/performance и optional features отложены.
В этой актуализации модель, тесты, Verify, Preflight и GPU probes не запускались;
source/runtime/release records и прежние evidence не менялись.
