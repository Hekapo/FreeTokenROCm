# Этап 6.1: portable source initialization Triton AMD knob

Дата: 12.10.2026. Ветка: `integration/research-refresh-20261002`.

Реализован опциональный source initializer `FREETOKEN_TRITON_CACHE_AMD_KNOB=1`.
Он вызывается при первом импорте `freetoken.kernel` в процессе, после загрузки Torch
и до kernel imports. CLI/daemon остаются без Torch до работы с kernels.

Сохраняется текущее effective bool `triton.knobs.amd.use_buffer_ops`. Поддерживается
точная проверенная форма `env_bool` Triton3.8.0 с ключом `AMDGCN_USE_BUFFER_OPS`;
другая версия/descriptor пропускаются без чтения неизвестного getter. Предыдущий
Python override уже считается cached; будущие explicit Python assignments действуют.
Инициализация запоминается по PID. Environment configuration меняется через fresh process.
Обычный default — flag off. Installed Triton files не меняются.

Публичный setter проверен по [исходникам Triton 3.8](https://github.com/triton-lang/triton/blob/release/3.8.x/python/triton/knobs.py):
assignment записывает bool в instance override и сохраняет его в environment.
Инициализатор не включает buffer ops принудительно: при effective=false сохраняется false.
Hook расположен в `python/freetoken/kernel/__init__.py`, implementation —
`python/freetoken/_triton_runtime.py`. Откат: убрать переменную либо задать `0`
и перезапустить процесс. Изменение переменной после импорта kernels не переинициализирует knob.

## CPU verification

28 tests passed, 10,80s: true/false, disabled flags, CUDA skip, versions/descriptor
compatibility, Python overrides и последующее assignment, invalid non-bool,
реальный kernel import в fresh process и daemon/CLI import safety. На реальном
descriptor после включения 1000 чтений вызывают environment getter один раз;
при off — 1000 раз. Новый test до implementation падал на отсутствующем helper.
В промежуточном fixture исправлен class-level descriptor access; production
использует static inspection и не вызывает getter при capability check.

## Frozen runtime comparison

Выполнены три phases: main / current source flag off / current source flag on.
Для каждого — один прогрев и два measured requests; одинаковый Python3.12,
AMD Torch2.13.0+rocm10.0.0, SDK10.0.0, Triton Windows3.8.0.post28 и все73 packages.
Source10.1/Torch2.14 candidate этапа5 не используется.

Профиль: Qwen3.6-35B-A3B-Q4_1, configured64K/BF16 KV, FP32 GDN, prefill512,
MoE2048/LRU, serial expert load, concurrency1, graphs0; buffer_ops=true.
Actual prompt34 tokens, temperature0, thinking=false, output budget512,
integers1..100 separated by commas. Renderer SHA256 совпал на main/current:
`1f87ebe867f592fbfbbf5a1933f5de075c56cd8a26b0872743ee4b375a3cf504`.

Проверка выполняется через managed observer/owned Stop, launch free commit45GiB
и прежние runtime floors12/4/3GiB. Profilers, finite diagnostics, token tracing,
scheduler stack monitor и lab knob pinning выключены. Каждый запуск использует
отдельный native/Triton cache; source/package/protocol manifests заморожены.
Сохранена обязательная safety проверка logits every64 для capacity>8K; её режим
одинаков во всех трёх phases. Полностью свободного от этого контроля A/B нет.

### Результат

| Source / флаг | Decode TPS, медиана | Два measured TPS | TTFT, медиана, s | Total, медиана, s |
|---|---:|---|---:|---:|
| main, `c4832f35055c` | 52,32 | 52,03 / 52,61 | 0,7575 | 6,3195 |
| current, off | 51,82 | 52,46 / 51,17 | 0,7660 | 6,3830 |
| current, on | 51,73 | 51,88 / 51,59 | 0,7735 | 6,3985 |

ON относительно OFF: **−0,16%**; ON относительно main: **−1,12%**.
Ускорение в этой задаче не подтверждено; различие ON/OFF меньше разброса двух
measured requests в каждом варианте. Это небольшая последовательная выборка
main→off→on, без чередования порядка; её недостаточно для вывода о небольшой регрессии.
OFF/main включает предыдущие изменения этапов1–5 и не изолирует этот initializer.
Исторические +27,44% не являются новым результатом.

Каждый из девяти успешных запросов дал 292 completion tokens, одинаковые bytes ответа,
пустой reasoning и `finish_reason=stop`. Content SHA256:
`ac4aac48ff9202a199cf471ac799946e83818fd9028b92d46ad998c89d2e1328`.
Во всех phases проверены capacity/KV65536, неизменный instance ID и idle после запроса.

Метрика decode TPS: `(completion_tokens − 1) / (finish_event_s − first_output_s)`.
Первый фрагмент `'1'` подтверждён как один token; его время и token исключены,
tail до события завершения включён. Client monotonic clock имеет около 15,6ms
дискретности на этом Python3.12/Windows. По два measured запроса после одного
прогрева; это короткий workload при configured64K, **не decode после заполнения 64K**.
Новых matched4K/8K замеров в этом этапе нет; исторические числа имеют другую задачу.

### Ресурсы и остановка

Минимумы observer за весь запуск, включая загрузку/прогрев:

| Phase | Free RAM, GiB | Free commit, GiB | Free local VRAM, GiB |
|---|---:|---:|---:|
| main | 25,261 | 13,127 | 6,888 |
| off | 25,275 | 13,136 | 6,888 |
| on | 25,315 | 13,183 | 6,888 |

Все значения выше runtime floors12/4/3GiB. Все три successful runs завершились
`STOPPED`: owned root/observer отсутствуют, process tree0, listeners0,
`recovery_required=false`. После последнего Stop: RAM50,500 / commit47,451 /
local VRAM15,108GiB. Исторический quarantine64K сохранён.

| Phase | Raw run |
|---|---|
| main | [171731-a5cf59a4](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/ctx64k-20261012-171731-a5cf59a4) |
| off | [172050-6425fcb6](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/ctx64k-20261012-172050-6425fcb6) |
| on | [172518-3499fdff](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/ctx64k-20261012-172518-3499fdff) |

### Исправления стенда до принятого сравнения

1. Попытка убрать периодическую проверку logits остановилась на существующем
   preflight gate capacity>8K, до запуска модели. Сохранён одинаковый `every64`.
2. В первом main startup стенд передал `FT_DIAG_SCHED_STACK_INTERVAL_S='0'`.
   Worker требует положительный interval, когда переменная задана; он завершился
   с `ValueError`, до тестовых запросов. Исправлено на отсутствие переменной.
   Все записанные owned PID и listener проверены как отсутствующие; только после
   этого устаревший STARTING/fault архивирован с отдельным review. Неудачный run
   сохранён, прежний quarantine не снят. Это Python startup error, не GPU hang.
3. Следующий main client не имел обязательного `stats_poll_s` и упал с `KeyError`
   после отправки warmup. Owned Stop прошёл. Spec дополнен значением0,5,
   required keys и весь client contract проверены офлайн перед следующим запуском.

Эти попытки не входят в timings/correctness таблицы. Журналы `main-run*.txt`,
`client-contract.json` и `main-startup-recovery/review.json` сохранены в audit.
Collector исправлен для фактического observer kind `resources`; минимумы получены
из сохранённых samples, новых model runs для этого не потребовалось.

### Решение и границы

Функциональная часть 6.1 завершена: CPU/import contracts и bounded model smoke
прошли. **Default остаётся off; ускорение не принято как доказанное.** GPU logs
не содержат отдельного INFO события initializer. Фактическое assignment/getter
caching подтверждено fresh-process
тестами реального descriptor; model phases используют замороженные source/flag manifests.

Control packages и installed files сохранены; после runtime изменены только
документация и внешние audit scripts. Исходники accepted runtime зафиксированы
до этих document edits. Нет нового long64K run и изменений compiler/native sources
в этом этапе. Stage5 AMD10.1/Triton post29 по-прежнему не принят из-за compiler failures.
К 6.2 переходить после доказательства одного расхода на текущем active path;
повторять широкую серию ради небольшого различия этой таблицы не требуется.

Raw evidence: [stage6 audit](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/stage6-20261012),
[comparison JSON](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/stage6-20261012/runtime-comparison.json).
