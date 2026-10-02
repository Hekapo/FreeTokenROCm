# План интеграции результатов исследования от 02.10.2026

**Статус: Git-подготовка выполнена, интеграция запланирована.** Найденные исследованием изменения ещё не перенесены и не проверены на нашей модели. Ветка для реализации: `integration/research-refresh-20261002`, создана от обновлённого и опубликованного `main` собственного форка после commit/push `64K-Migration`.

### Зафиксированная база

| Ссылка | Фактическое состояние |
|---|---|
| Собственный форк | [Hekapo/FreeTokenROCm](https://github.com/Hekapo/FreeTokenROCm) |
| `main` до слияния | `a50984cf2d5fa8acbe0f1d91cb8dfe74b508aae8` |
| Commit 64K | [`c4832f35055c5f27998fd36996d21d4dbda18b76`](https://github.com/Hekapo/FreeTokenROCm/commit/c4832f35055c5f27998fd36996d21d4dbda18b76), `feat(context): checkpoint 64k migration and diagnostics` |
| `main` после fast-forward | `c4832f35055c5f27998fd36996d21d4dbda18b76`; remote SHA подтверждён после push |
| Начало новой ветки | Тот же `c4832f35055c5f27998fd36996d21d4dbda18b76`, после успешной публикации `main` |
| Сохранённый статус 64K | [64k-migration-status.md](64k-migration-status.md): известный ordinary smoke FAIL, 133 CPU tests до исправления shell и 12 shell tests после него |

В checkpoint также исправлена найденная при review ошибка shell: начальная эффективная ёмкость 8K больше не ограничивает рост отображаемого бюджета до 64K после обновления KV. Модельные прогоны на Git-этапе не выполнялись.

## 1. Цель и исходное состояние

Выбрать применимые исправления и ускорения для **native Windows, RX 9070 XT / gfx1201, Qwen3.6-35B-A3B с весами Q4_1 и KV BF16**, переносить их отдельными проверяемыми изменениями. Сначала восстановить корректность обычного длинного prefill, затем переносить подтверждённое устранение CPU-затраты и измерять новые оптимизации. Обновление библиотек проводится независимо от функциональных изменений.

Исходная исследованная ветка: `64K-Migration`, HEAD `a50984cf2d5fa8acbe0f1d91cb8dfe74b508aae8` плюс сохранённые локальные изменения. Активный стек: Python 3.12.0, AMD Torch `2.13.0+rocm10.0.0`, ROCm 10.0.0, Triton Windows `3.8.0.post28`, TVM FFI `0.1.13.post3`, драйвер Adrenalin 26.8.1. Система Windows 11 Pro build 26100 / 24H2; опубликованная матрица ROCm 10 валидирована для 25H2. Это ограничение поддержки, а не установленная причина отказа.

Профиль модели: KV 65536, naive cache, graphs 0, MoE cache 2048 / LRU, serial experts, backend/frontend 1, prefill 512. Q4_1 — формат весов; изменение формата KV будет отдельной работой. Основные места кода в плане указаны относительно корня репозитория.

### Честная граница готовности 64K

- Историческая серия прошла 602/602 запроса и 8 clean Stops; её runtime сохраняет собственный положительный результат.
- Отдельный инструментированный A/B process-start caching настройки Triton дал 40,7111 → 51,8806 токен/с (+27,44%), 62/62 запроса. Это подтверждённая затрата, а не доказательство устойчивости всех будущих запусков.
- Последний обычный ускоренный smoke: короткий запрос 51,975 токен/с; длинный запрос завершился NaN/Inf после 4096 завершённых prefill-токенов, `phase=prefill forward=253`. Третий запрос не выполнялся, release record не создан, recovery marker сохранён.
- Причины этого NaN, прежнего зависания ПК и постоянной просадки до 30 токен/с не установлены. Слияние 64K в `main` сохраняет разработки и эту известную проблему; оно не даёт новый статус квалификации ускоренного runtime.

## 2. Что из исследования сохраняем в репозитории

Срез охватывает 82 пакета; 1412 видимых форков, 3529 веток; 51 впервые учтённое имя и 9 дополнительных полных копий движка; 27 внешних проектов. Составлены 40 кандидатов из FreeToken и 28 из внешней экосистемы. Подробно прочитаны выбранные применимые patches, а не каждый из 1279 обнаруженных недавних commits.

В Git достаточно сохранить этот план и компактное описание baseline/результатов/известного отказа. Огромные raw API dumps, wheels, модели, бинарные kernels, полные локальные логи и пути к персональным окружениям остаются в локальном архиве исследования. Для будущих переносов ниже сохранены точные donor SHAs, первичные ссылки и условия применимости; интеграция не зависит от публикации raw-архива.

Локальный архив: `rocm10-20260926/research-refresh-20261002/` в workspace. Основные файлы: `SUMMARY_RU.md`, `libraries/versions.json`, `forks/candidates.json`, `ecosystem/candidates.json`. Это снимок на 02.10, а не вечный lockfile. Перед установкой позднее следует повторно проверить только выбранные wheel URLs/hashes/constraints и документацию используемых API.

## 3. Общие правила реализации и проверки

1. Один смысловой кандидат — отдельный коммит. Переносить необходимые hunks с сохранением attribution; не сливать целиком чужие экспериментальные ветки. В сообщении и отчёте фиксировать donor SHA, локальный SHA, активный путь, shape/dtype и результат.
2. На этапе функционального исправления сохранять исходные packages, SDK, driver, модель, chunk/cache settings и режим Moonlight. Обновление стека, dtype, cache lifecycle и kernel scheduling не смешивать в одном сравнении.
3. Применять уже существующие правила `docs/rocm-experiment-baseline.md`: отдельные native build/cache directories, проверенный imported source path, Torch до Triton, один сервер на GPU, explicit offload strategy, graphs 0. Cache roots — разные абсолютные директории для разных окружений и кандидатов. Venv не изолирует driver.
4. Сохранять memory/watchdog/fault/managed Stop guards. Текущий startup gate свободного commit 45 GiB и runtime RAM/commit/local VRAM 12/4/3 GiB не ослаблять ради старта. Recovery marker меняется только через существующую проверенную процедуру; успешный Git merge или unit test его не снимает.
5. Проверять изменённый контракт, а не дублировать реализацию. Численная эталонная проверка нужна для masks, GDN state, quant math и GEMV; для документации и package metadata достаточно статических проверок.
6. Учитывать просьбу о меньшем числе тестов: сначала offline/source checks и небольшой operator fixture; затем ограниченный модельный smoke. Серия 602 запроса, длительный soak и перебор всех settings не являются обязательным этапом этого плана.
7. Первый NaN/Inf, ошибка device, watchdog, memory guard или неудачный Stop завершает попытку. Не запускать следующий вариант автоматически; сохранить evidence и разбирать узкое воспроизведение. Повтор оправдан новой конкретной гипотезой или исправлением.

### Небольшие бюджеты

| Проверка | Предельный стартовый объём | Что подтверждает |
|---|---|---|
| Новый kernel/math контракт | Один компактный параметризованный fixture; только затронутые shapes, edge cases и CPU reference | Корректность конкретной операции |
| Локализация отказа | Один managed запуск, readiness + короткий запрос + один запрос с прежним длинным вводом; остановка на первом non-finite | Первый обнаруженный нарушенный контракт; не полная квалификация |
| Smoke после исправления | Readiness + последовательность short → прежний long → short, один managed Stop | Возврат обычного пути без прежнего отказа |
| Оптимизация одного operator | Один warmup и 3 замера на вариант; GPU timing отделён от model timing | Предварительный эффект на операции |
| Перенос подтверждённой оптимизации | До 6 коротких запросов A/B в одном согласованном порядке; длинный smoke только при изменении correctness/runtime риска | Локальная воспроизводимость; при шуме вывод «не определён» |
| Итоговая скорость по контекстам | По 2 одинаковых коротких запроса на 4K/8K/64K, затем один 64K long при необходимости | Новый сопоставимый небольшой срез, не soak |

Это лимиты первой попытки, а не требование выполнить каждую строку для каждого коммита. Готовый fixture повторно используют, а модельные smoke объединяют для совместимых изменений только после раздельных offline gates. Если имеющегося evidence достаточно, дополнительные модельные проверки не добавляются.

## 4. Порядок этапов

### Этап 0 — сохранить 64K и создать точку начала

**Выполнено 02.10.2026 до реализации кандидатов.** Ниже сохранён порядок выполненных операций.

1. Просмотреть локальный diff, новый тест, repository instructions и текущие remotes. Проверить, что в коммит входят source/doc изменения 64K, а не модели, raw evidence, venv или секреты.
2. Закоммитить локальные изменения на `64K-Migration`, push в собственный форк. Зафиксировать точный SHA и известный обычный smoke FAIL в кратком сохранённом отчёте.
3. Сопоставить локальный и удалённый `main`, сохранить его новые изменения, merge `64K-Migration` в `main` собственного форка и push `main` без переписывания истории.
4. Только после успешной публикации создать `integration/research-refresh-20261002` от точного нового `main`. Сохранить этот план в `docs/research-integration-plan-20261002.md`; запись о базовом SHA должна соответствовать выполненным операциям.

**Приёмка выполнена:** удалённые SHA `64K-Migration` и `main` подтверждены; 64K changes доступны в истории; новая ветка создана от этого `main`; первоначальные локальные изменения сохранены и known failure записан. Это Git-этап без новых inference runs.

### Этап 1 — локализовать non-finite и проверить границы данных

**Первый этап реализации.** На старом стеке добавить opt-in диагностику, затем переносить только подтверждённые нарушения masks/padding/ownership.

**1A. Первый неконечный тензор.** В `python/freetoken/models/qwen3_5_moe/gdn.py`, `gdn_kernels.py` и границах GDN/MoE операций регистрировать finite-check на выбранном участке prefill: номер layer/chunk/forward, shapes, dtypes, state indices/axes, kernel specialization/autotune key. Снимок ограничить одной сбойной операцией и её непосредственно предшествующими входами, суммарно не более 64 MiB; не сохранять всю модель или KV. Сначала локализовать границу, затем подробный snapshot именно там. GPU synchronization и перенос в CPU могут менять timing; это диагностический режим, его скорость не сравнивать с обычной.

**1B. GDN partial tiles / zero padding / int64 адреса.** Донор [FLA #1062](https://github.com/fla-org/flash-linear-attention/pull/1062), branch revision `cfab1960f257bd083e3da00cceb66dd64c885692`, merge `02e270330b4fc5bb2fd738281b5d594bc89e6d5f` (25.07). В локальных `kernel/fla/chunk_delta_h.py`, `chunk_fwd.py`, `chunk_o.py` проверить все `tl.load`/block pointers с boundary check: нейтральное значение padding должно быть явным до dot/exp/reduction. Mask после умножения не устраняет `NaN * 0`. Проверить вычисление offsets в int64 там, где диапазон способен превышать int32. Сохранить наш indexed state `[V,K]`, GQA и strides; upstream `[K,V]` не заменяет местный контракт.

**1C. LDS guards.** Сравнить локальный pruning с [FLA #751](https://github.com/fla-org/flash-linear-attention/pull/751), revision `85856e259f6184bb5f16dcf686b2fea41b6b7d34`, merge `bbdd5051aea72021a35d7a9dde3c03dc9752ba69`. Перенести только недостающие корректные ограничения для 64 KiB LDS gfx1201. Не фиксировать BK=32 лишь потому, что он был выбран в успешном запуске: все шесть изолированных KKT settings уже прошли прежнюю CPU-double проверку, причинность autotune не установлена.

**1D. Физический native padding и lifetime.** Проследить allocation → copy → native read range для expert images в `models/qwen3_5_moe/gguf_experts.py`, `moe/offload_cache.py`, `moe/offload_kernels.py`, `kernel/csrc/gguf/{moe,moe_vec,mmq,mmvq,vecdotq}.cuh`. Идея инварианта из [RDNA r31](https://github.com/stew675/llama-cpp-rdna-boosts/releases/tag/v16-84e76d8a2-r31), точный release commit `76a083ed492756803187c85de9ce4537a787b606`: выделенная физическая область должна покрывать максимальный kernel read и иметь finite/zeroed tail. Проверить именно Q4_1 layout, affine minimum term и края последнего expert/row/block. Donor Q4_K/Q8/IQ4 kernel нельзя переносить как Q4_1.

Для pinned H2D sources отдельно проверить срок жизни staging buffer до завершения соответствующего stream/event, отсутствие reuse до completion, ownership metadata. Источник гипотезы: [ROCm #12535](https://github.com/ROCm/rocm-systems/issues/12535), reproducer SHA `b340cc77350f58b1fc116c62b767867e67c862a8` из `zspitzer/rocm-windows-sdma-hang-gfx1100`. У него gfx1100, Torch 2.13 / ROCm 10.1 и first-use hang; у нас gfx1201 и NaN. Использовать описание для аудита; исходники без найденной лицензии не копировать. `PAL_DISABLE_SDMA=1` не включать как default. Короткий изолированный A/B этого флага допустим позднее только при evidence о copy/SDMA пути, с прежними guards и фиксацией всех параметров.

**Приёмка:** определена первая обнаруживаемая non-finite граница или сохраняется честное «не локализовано»; подтверждённые fixes имеют ограниченный reproducer. Fixture включает фактические model shapes, частичные T вокруг границ chunk/tile, непустое начальное состояние, varlen/indexed state, NaN-poison вне допустимого диапазона. Адресные проверки для больших offsets делаются без гигантских allocations. CPU double/FP32 reference и numerical tolerance задаются до сравнения. Если патч нужен только теоретически, он оформляется как защитный invariant, а не как доказанная причина прежнего отказа. Обычный long smoke проходит отдельно от instrumented capture.

**Откат:** revert отдельного патча, прежний stack/cache root; диагностика выключена по умолчанию. При отрицательном smoke fault marker сохраняется, этап 2 остаётся закрыт.

### Этап 2 — перенести process-start caching Triton knob в source

**Зависимость: этап 1 дал обычный корректный длинный smoke.**

Это локальный подтверждённый кандидат, сейчас реализованный в лабораторных `stable_triton_knob.py`, `triton_knob_control.py`, `cached_stats_launcher.py`, а не завершённый portable source feature. Результат сохранён в `results/ctx64k-knob-compare-20261002/REPORT_RU.md`. Перенести минимальный инициализатор в repository source path и вызывать один раз в каждом Python process до первых kernels. Сначала импортировать Torch/SDK, затем Triton. Прочитать эффективный bool `triton.knobs.amd.use_buffer_ops` и публичным assignment закрепить **то же значение**, не принудительно `True` и не отключать buffer ops.

Первоначальная активация только opt-in на проверенном AMD Triton 3.8 runtime, до успешного собственного source smoke. Версия сама по себе не заменяет correctness gate. Явно описать process-start semantics: поздняя смена environment не меняет pinned getter; для смены настройки нужен новый процесс. Сохранять existing explicit override; не перезаписывать неизвестные версии/runtime descriptor. Имя нового source flag определить при реализации рядом с существующими config conventions; не вводить выдуманное API в инструкции до реализации. Диагностическая обратимость A/B остаётся opt-in; профилировщик и backend telemetry в обычный source не переносятся.

**Приёмка:** CPU tests actual bool true/false, existing override, unsupported version, once-per-process и original mode; короткий plain-runtime smoke с прежним long. Сопоставить прежде выбранные package/source hashes и фактический импорт. Если дополнительная скорость измеряется — малый A/B, без cProfile, одинаковые prompts/usage и knob value. Старые +27,44% не объявлять новым результатом.

**Откат:** flag off / revert и fresh Python process; runtime cache namespace и исходное чтение environment восстановлены. Не редактировать установленный Triton in place.

### Этап 3 — отдельно проверить согласованный AMD RC стек

**Обычная зависимость:** сохранён source baseline после этапов 1–2 и его результаты. В рамках той же интеграционной ветки отдельный коммит constraints/config; отдельные venv, native build и caches. Рабочую среду ROCm 10.0 не заменять.

**Диагностическая развилка из этапа 1:** если конкретный failing operator или DLL/compiler evidence указывает на SDK/Triton, RC можно проверить до исправления старого runtime. Заморозить точный source SHA, knob mode и остальные настройки; выполнить static gates и один ограниченный smoke. Успех или отказ такого сравнения фиксирует диагностический результат. Перенос ускорений и переключение default остаются закрыты до ordinary correctness gate; устранение старого runtime не является условием самого RC diagnostic.

| Вариант | Состав | Цель |
|---|---|---|
| E0 control | ROCm 10.0.0 + AMD Torch 2.13.0 + Triton post28 | Сохранённый контроль |
| E1 SDK isolation, условный | Согласованный ROCm `10.1.0rc3` + AMD Torch `2.13.0+rocm10.1.0rc3` и matching device wheels; сохранить Triton Windows `3.8.0.post28` | Только если требуется отличить эффект SDK/vendor Torch от смены Torch API; это наш экспериментальный stack, не заявленная vendor pairing |
| E2 основной RC candidate | ROCm `10.1.0rc3` + AMD Torch `2.14.0+rocm10.1.0rc3` + matching gfx1201/device family wheels + Triton Windows `3.8.0.post29` | Проверить опубликованную пару Torch 2.14 / Triton 3.8; сравнивается весь комплект, не один фактор |

Индексы: [AMD RC Torch](https://rc.repo.amd.com/rocm/whl-next/torch/), [AMD RC SDK core](https://rc.repo.amd.com/rocm/whl-next/rocm-sdk-core/). Selector `rocm` и SDK core/devel/libraries/device-gfx1201 выбираются как набор `10.1.0rc3`; Torch и обе `amd-torch-device-gfx1201` / `amd-torch-device-gfx12-0` distributions имеют одинаковую выбранную vendor Torch версию. В каталоге AMD RC сохранены version/filename/URL, но хеши этих бинарных wheels ещё не получены. Перед будущей установкой скачать выбранные файлы в отдельный каталог, получить и записать SHA256/metadata, затем проверить полный разрешённый manifest. Не выбирать per-package maximum. Обычный upstream PyPI Torch 2.14.1 и nightly SDK 10.2 / Torch 2.15 не заменяют этот набор. E1 не является обязательным дополнительным запуском: его используют только для диагностической развилки.

Перед установкой: проверить CPython 3.12 Windows tags, constraints/build-system pin `<2.14`, собственные Torch internal API/native extension ABI и DLL paths; подготовить полный разрешённый manifest. Нужные source bounds менять точечно и обосновывать. Установить только в новую среду, затем metadata/import/native/JIT probe и ограниченный smoke. Зафиксировать actual SDK/compiler/DLL source, графы выключены. GPU/runtime checks выполняются после static gates, а не при сборе metadata.

**Приёмка:** resolved manifest, корректные native builds, actual gfx1201, импорт нужного checkout, обычный short/long/short и managed Stop. При отказе сохраняется старая среда; RC не становится default. Публикация vendor Windows wheel не считается доказательством нашей совместимости.

**Условный hipBLASLt кандидат:** [ROCm/rocm-libraries #11340](https://github.com/ROCm/rocm-libraries/pull/11340), revision `bf725a806a5215bc9eb131c23da8f31123076ab9`, merged 02.10. Проверять наличие fix в выбранном SDK и применение `getAllSolutions` конкретным активным math path. Порядок solutions hipBLASLt и Triton KKT autotune — разные механизмы. Не собирать свою глобальную hipBLASLt DLL и не считать её причиной NaN без call-path evidence.

### Этап 4 — минимальные обновления библиотек и diagnostics

**Зависимость:** выбран один принятый stack; source feature изменения на время package comparisons заморожены. Не обновлять все 82 пакета вместе.

| Группа | Кандидат | Gate и порядок |
|---|---|---|
| Diagnostics/build | Triton Windows `3.8.0.post28` → `post29`, release SHA `143426c4cbead9d3baf826b017ce1d3da5e51d3c` | [Release](https://github.com/triton-lang/triton-windows/releases/tag/v3.8.0-windows.post29). Только если выбранный stack ещё на post28; после E2 повторный перенос не нужен. LLVM remarks/error/MAX_JOBS changes; AMD backend fix отсутствует. Отдельная среда/кэши, metadata и один compile probe; model smoke совмещать с уже нужным stack gate |
| Native FFI | TVM FFI `0.1.13.post3` → `0.1.14.post1`, source SHA `e251c6abee7a111c6467f84d854be512e33e2db0` | [Release](https://github.com/apache/tvm-ffi/releases/tag/v0.1.14-post1). Сначала согласовать точный source pin и binary/API/native extension ABI; source SHA не заменяет wheel hash |
| Model tooling | Transformers `5.18.0` + HF Hub `1.33.0` | Внутри текущих core bounds; обновлять вместе. Проверить tokenizer/config/model metadata нашего GGUF, не менять checkpoint |
| CPU JIT | Numba `0.68.0` + llvmlite `0.50.0` | Совместная dependency group. При отсутствии активного использования кандидат переносится вниз приоритета; нужный CPU JIT path проверяется отдельно |
| Packaging completeness | ModelScope source requirement отсутствует в active env | Сначала доказать нужен ли он активному GGUF path, затем либо корректно установить для выбранного supported entry point, либо сделать зависимость optional; не добавлять тяжёлые dependencies для формы |
| API dependencies | FastAPI/Pydantic и остальные bounded updates | Только при конкретной пользе. У FastAPI `0.142.2` новая обязательная OpenTelemetry API зависимость; Pydantic-core обновляется строго с Pydantic, а не отдельно |

Релиз TVM `0.1.14.post1` опубликован 22.09: он впервые учтён сейчас, но не новый после 26.09. PyPI wheel URLs и опубликованные SHA256 берутся из `libraries/versions.json` и перепроверяются перед установкой; для AMD vendor RC хеши фиксируются отдельно по процедуре этапа 3. Commit исходников не заменяет хеш binary artifact.

Upstream NumPy 2.5.3, OpenAI SDK 3.24.0, HF Hub 2.1.1 выходят за текущие core bounds — отложить их major/bounds migration. `flashlib 0.3.0` остаётся актуальным, менять нечего. Прямо сейчас не менять driver, Windows version или Python ABI. Обновления Git/uv и source-only Python 3.12.15 security release оформить позднее как отдельное обслуживание, не как исправление GDN; Python 3.14 из PATH не является новым interpreter для active cp312 wheels.

**Приёмка:** manifest с constraints/hashes и dependency checks, выбранный импорт/entry point, smoke только при затронутом активном runtime. Один отказ откатывает одну dependency group к сохранённому manifest; оставшиеся группы не наслаиваются на него.

### Этап 5 — ускорить короткий prefill и decode по фактическому bottleneck

**Зависимость:** обычный long корректен и stack зафиксирован. Порядок внутри этапа определяется коротким профилем/операторными timings, не популярностью репозитория.

| Кандидат | Точный источник | Что переносим и критерий |
|---|---|---|
| Routed short-prefill experts | [YevheniiKotyrlo `ff4966bd3b57de03f1163967d6c07de2a41d7ee9`](https://github.com/YevheniiKotyrlo/FreeToken/commit/ff4966bd3b57de03f1163967d6c07de2a41d7ee9) | `ensure_experts`/slot remap только для реально routed experts, adapt Q4_1 native bank; opt-in threshold. Сверить outputs, cache eviction, duplicate IDs, capacity, TTFT на short и 512-token chunk. RTX3090Ti/NVFP4 donor gain не переносится как наш результат |
| Bounded MoE workspace | [tomasuz `196eace47a9df5937113274a61fa459a594bb8fd`](https://github.com/tomasuz/FreeToken/commit/196eace47a9df5937113274a61fa459a594bb8fd) | Sub-batches после admission experts; у нас prefill 512 уже ограничивает обычный chunk, поэтому сначала доказать дополнительную пользу. Не создавать лишние launches без необходимости |
| Dense BF16 GEMV | [YevheniiKotyrlo `fdf015aeed5f30b1c89b9fea51e41ff2bc5993a0`](https://github.com/YevheniiKotyrlo/FreeToken/commit/fdf015aeed5f30b1c89b9fea51e41ff2bc5993a0) | Biasless, single-row, FP32 accumulation, width guard. Audit `layers/quantization/linear/unquantized.py` и actual GGUF dense dispatch: F32 GDN slots могут bypass registry. Численная сверка + near-tie router top-k, speed только на actual widths |
| Shared activation quant / router GEMV | [KerchumA222 `0c8c1853`](https://github.com/KerchumA222/FreeToken-ROCm/commit/0c8c1853), [router `a8044ae5`](https://github.com/KerchumA222/FreeToken-ROCm/commit/a8044ae5) | Older separate ideas; перед портом разрешить полный SHA из сохранённой branch/tree history. `_packed_runs` merge уже есть, `_shared_q8` нет. Native packed layout/affine Q4_1/near-tie routing требуют собственного fixture |
| GDN views / Conv1D fusion | [SGLang #38806](https://github.com/sgl-project/sglang/pull/38806) `17f89f31cefde8bccdcfe3a875cc4c5446f34e6b`; [#41677](https://github.com/sgl-project/sglang/pull/41677) `b6ef2239c999fc7b5a0165573807d9b05fd34880` | Только remaining copies/materialization и Conv1D fusion. Fused projection/GQA/l2norm уже есть. MI355X/CDNA launch geometry не переносить напрямую |
| Sync/metadata overhead | [SGLang #41762](https://github.com/sgl-project/sglang/pull/41762) `4f03ad15a5cc1a47b2e72992e2da774ede8e9715` | Closed unmerged proposal; изучить scalar assignments/index_fill и pinned ownership. Сначала lifetime audit этапа 1; убрать sync без сохранения lifetime нельзя |

Дополнительные low-priority native идеи: gfx1201 grid-stall guard [RDNA #78](https://github.com/stew675/llama-cpp-rdna-boosts/pull/78) `536a7b68f436861bd3d2d3404e7a1a311eeefc9e` применим только после доказательства аналогичного grid/body в Q4_1; non-temporal copy, sudot4 и two-row Q4_0 → Q4_1 требуют отдельной реализации/byte or numerical parity. Эти идеи не включаются автоматически.

**Приёмка оптимизации:** корректный active path, no extra NaN/Inf, bounded VRAM/host/pinned workspace, отсутствие ухудшения обычного длинного smoke при затронутой ветке. Сохранить TTFT, decode TPS, wall time и условия; вывод о выигрыше делать только если он больше разброса контрольных замеров. Неэффективный кандидат остаётся default-off либо откатывается. Чужие 2–4x gather-prefill числа r31 отозваны автором из-за NaN routing и исключены из целей.

### Этап 6 — точность и bounded prefix reuse

**Зависимость:** cold naive обычный long корректен; это новая функция radix-prefix profile, она не нужна для восстановления текущего naive cold-prefill.

1. Сначала workspace planner: [gdevenyi `081e50836ebc746e604ac52d6986fc996bab924f`](https://github.com/gdevenyi/FreeToken/commit/081e50836ebc746e604ac52d6986fc996bab924f) и rebuild-consistency [`5286dd08b74dce666b2a2668c8975d83d6ef4437`](https://github.com/gdevenyi/FreeToken/commit/5286dd08b74dce666b2a2668c8975d83d6ef4437). Считать реальные h/v_new и staging dtypes; startup и rebuild используют одинаковый reserve. Reserve проходит до увеличения FP32 allocations.
2. FP32 промежуточное h: [JUNQINGV587 `0a3e111a198a2a12c6324ab4cf3eebd3b2cdd41a`](https://github.com/JUNQINGV587/FreeToken/commit/0a3e111a198a2a12c6324ab4cf3eebd3b2cdd41a), cast в q dtype только у output dot. `chunk_delta_h.py` сейчас создаёт h по dtype k; поздний cast в FP32 recurrent pool не восстанавливает утраченную точность. Это удваивает BF16 h workspace, поэтому отдельно измерить bytes/fit decision. Не приписывать этому fix причины cold NaN.
3. Final short chunk state: [ServeBig-project `481994fe71cc4483e19cd54c77ee726f714a99da`](https://github.com/ServeBig-project/FreeToken/commit/481994fe71cc4483e19cd54c77ee726f714a99da). Адаптировать к нашему ping-pong snapshot-slot lifecycle; cache length и start snapshot нельзя просто скопировать из donor.
4. После state-parity gate — bounded pinned host bank: [YevheniiKotyrlo `b1fd3626a1ca9cf1483c14a7a523ba1fe2b8c691`](https://github.com/YevheniiKotyrlo/FreeToken/commit/b1fd3626a1ca9cf1483c14a7a523ba1fe2b8c691) + reclaim [`6036cbd3e6650f1c75eaa661a3fed64508a9ff12`](https://github.com/YevheniiKotyrlo/FreeToken/commit/6036cbd3e6650f1c75eaa661a3fed64508a9ff12). Archive только frozen intermediate states, строгие slots/bytes bounds, events/ref ownership, освобождение после evict/Stop. Default 0 slots; рассчитывать допустимые slots из текущего host/commit reserve. Donor 128 slots / 13,79 GiB pinned RAM не является нашим default.

**Приёмка:** компактный cold-vs-resume numerical/state fixture на chunk boundary и final partial chunk, nonzero initial state, eviction/rebuild, отсутствие aliasing и pinned growth. Затем один длинный prefix и одна continuation в отдельном opt-in radix profile. Сравнивать quality/logits, TTFT reused prefix и memory с cold baseline. Не включать одновременно полный host KV/disk tier; это иной lifecycle.

**Откат:** naive profile, host slots 0, fresh process; отдельные commits planner/precision/lifecycle позволяют убрать функцию без отката correct masks.

### Этап 7 — компактный итоговый срез и решение о default

Зафиксировать один принятый runtime manifest. На одинаковой короткой задаче провести по 2 запроса с configured 4K/8K/64K, одинаковым output budget, cache policy, процессной настройкой knob, драйвером и Moonlight state; token counts/KV allocations записать отдельно. Выполнить 64K long и Stop только если это ещё не подтверждено тем же exact runtime. Указать малый sample size; не объявлять длительную устойчивость из нескольких запросов.

Старые reference short medians 4K 40,05 / 8K 40,27 / 64K 41,40 TPS сохранить как historical. Новый 64K показатель сравнивать с новыми 4K/8K из этой же серии; не делить single cached smoke на исторические числа как парный результат. Отчёт содержит correctness, TTFT, decode TPS, memory minima, plain-vs-instrumented status, clean Stop и source/package/runtime hashes.

**Готово:** принятые кандидаты указаны с результатами и откатами; неприменимые и невыгодные отмечены; новая branch clean и published по отдельному пользовательскому поручению. Обычный profile/release gate меняется только после успешной проверки exact runtime; известно, что merge 64K раньше этого этапа сам по себе ничего не снимает. Длительный TTFT (~93 с на прежнем 63K instrumented run) — отдельный показатель; целевого времени пользователь пока не задавал.

## 5. Очередь отложенных и исключённых кандидатов

| Кандидат | Решение и основание |
|---|---|
| Safe fully masked sampling softmax | Небольшой будущий correctness fix [gdevenyi `f1dd670dc46e6e9f92e6919a5f7053c49db2f3fd`](https://github.com/gdevenyi/FreeToken/commit/f1dd670dc46e6e9f92e6919a5f7053c49db2f3fd), но только stochastic/min_p/min_tokens path. Не лечит активации GDN и greedy; ставить отдельным коммитом/fixture после этапа 1, когда этот режим востребован |
| Triton RDNA barrier #10906 и BF16 #11227 | Не содержатся в release post28/post29. Barrier уже отдельно пробовали для старого alignment без успеха. Custom compiler fork — самостоятельный эксперимент только при конкретном failing operator; переход post29 не выдать за эти fixes |
| Windows memory NVML fix | [YevheniiKotyrlo `249684680f3e268d4dc335b101f0767ea78b246d`](https://github.com/YevheniiKotyrlo/FreeToken/commit/249684680f3e268d4dc335b101f0767ea78b246d) возвращает прежний fallback на AMD; не интегрировать как ROCm budget fix |
| SSE heartbeat/cancellation, pinned attention, merged GGUF projections | Локальный эквивалент уже есть; повторный перенос не нужен |
| vLLM GGUF plugin refresh | Все 52 qualified заимствованных Triton runtime files совпали с head `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`. Нужна только отметка provenance; нет нового kernel patch для переноса |
| AITER full-attention #5790 | PR head `ab285548ed50c66f0e1e1862113fc7b74aac71b6`, merge `9cb26291f6daafc85bc65a3b85cc243bf73159ca`: учитывать 64KiB LDS, но donor padded heads 576/768/1024 не равны нашим D256. Нового backend dependency пока не добавлять |
| AITER Sage #5900 | Open; exclusions causal/GQA/masks делают его непригодным для нашего autoregressive attention |
| gfx1250 async-copy / FlyDSL gfx95 | SGLang #41868/#39595 target другой GPU; не включать guards/backend на gfx1201 по аналогии |
| Quantized KV: Octave / TurboQuant | Octave draft [#59774](https://github.com/vllm-project/vllm/pull/59774) `5934c7cf616277dfc8b1f20e97e828e46c10cdf2` native prefill пока gfx1100–1103; TurboQuant другой engine. Большая новая format/backend/quality работа после ordinary correctness; отдельно от Q4_1 весов |
| MTP / speculative replay | [KerchumA222 MTP `857b6bc88d01572e0ee52467bfd1a6641ee02ae3`](https://github.com/KerchumA222/FreeToken-ROCm/commit/857b6bc88d01572e0ee52467bfd1a6641ee02ae3), [SGLang replay #42209](https://github.com/sgl-project/sglang/pull/42209) `adb36489df91a885a1f2e85680dd6c86f35c3feb`. Нет нашего gains/correctness evidence, потребуется точный GDN rollback и acceptance semantics; graphs 0/offload transfer может ограничить пользу |
| CPU expert backend / predictive prefetch / disk tier | Изученные ideas сохранены; новый scheduling/ISA/ownership/backend scope. Не включать в первые этапы; no-license code не копировать |
| Desktop, driver, Windows/Python/tool upgrades | Отдельное обслуживание. Desktop beta.23 не исправляет kernel; driver 26.9.2 gaming freeze fix не доказан нашим freeze; Windows/Python changes не смешивать с NaN A/B |

## 6. Журнал выполнения

| Этап | Состояние на момент создания плана | Evidence/следующее действие |
|---|---|---|
| 0. Git сохранение/ветка | Выполнен | Commit/push 64K и fast-forward/push main: `c4832f35055c5f27998fd36996d21d4dbda18b76`; новая ветка создана после публикации |
| 1. Non-finite + masks/padding/lifetime | Запланирован, первый этап реализации | Offline path audit → bounded capture одного long отказа → узкий fix |
| 2. Source knob | Запланирован | Закрыт до ordinary correctness gate |
| 3. RC stack | Запланирован, отдельная среда; условный diagnostic из этапа 1 | Vendor wheel hashes перед установкой, совместимый manifest/API/native ABI gates |
| 4. Package groups | Запланирован, выбираются по пользе | No bulk upgrade |
| 5. TTFT/GEMV | Запланирован | Actual bottleneck, Q4_1 adaptation, small A/B |
| 6. Prefix reuse | Запланирован как opt-in новая функция | Workspace → precision → lifecycle → host bounds |
| 7. Итоговый срез | Запланирован | Exact runtime, малые 4K/8K/64K измерения |

**Первое поручение для реализации:** выполнить этап 1A и read-only audit 1B–1D на сохранённом стеке; внести только обнаруженное нарушение контракта отдельным патчем, приложить компактный reproducer. Для конкретной SDK/compiler гипотезы разрешена ограниченная диагностическая развилка этапа 3. Перенос performance features и принятие нового default требуют ordinary correctness gate. Этот план сейчас не запускает модель и не объявляет известный NaN исправленным.
