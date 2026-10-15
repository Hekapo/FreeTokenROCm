# 64K: сводный отчёт по сохранённым логам — 15.10.2026

## Результат

64K на ROCm 10.1 уже проходил длинные запросы, границу контекста, очередь,
отмены и длительную нагрузку. В полной серии 13 октября корректны 353 запроса,
Stop завершился чисто. При этом остаются два разных открытых вопроса:

- **Скорость после длительной нагрузки:** контроль снизился с 50,02 до
  39,06 токен/с. Позднее два обратимых переключения CPU affinity воспроизвели
  прирост 25–28%, но размещение потоков во время прежнего снижения не записывалось.
- **Остановка длинного запроса:** отдельный probe перестал показывать прогресс
  при счётчике prefill 54 272 из ожидаемых 63 488 токенов. Стек backend в этот
  момент отсутствует, причина не установлена, штатный Stop не был успешным.

Новая диагностика подготовлена и имеет сохранённый результат 118 CPU-проверок.
Модель с ней ещё не запускалась. Целевой short → long → short 15 октября
отложен по просьбе пользователя; результата этого прогона нет.

Этот отчёт составлен чтением существующих JSON, JSONL, XML и текстовых логов.
Новых запусков модели, тестов, Verify, Preflight, GPU probes или замеров здесь
нет. Таблицы памяти и интервал heartbeat неудачного probe пересчитаны только
из сохранённого `observer.jsonl`. Состояния Stop ниже исторические: живые
процессы в рамках подготовки отчёта не опрашивались.

## 1. Область данных и окружение

Основная область — модельные запуски и последующая диагностика 13.10.2026,
а также подготовленная диагностика и отменённая попытка 15.10.2026.
Время в хронологии — **Москва, UTC+03:00**; память — **GiB**.

| Параметр | Записанные условия |
|---|---|
| Модель | `Qwen_Qwen3.6-35B-A3B-Q4_1.gguf` |
| GPU / CPU | RX 9070 XT, gfx1201, 16 GiB / Ryzen 7 9700X, 8 ядер, 16 LP |
| ОС / драйвер | Windows 11 24H2, build 26100 / DriverStore 32.0.31041.1004 |
| Старый стек A/B | Torch 2.13.0 + ROCm SDK 10.0.0 + Triton Windows 3.8.0.post28 |
| Новый стек | Torch 2.14.0 + ROCm SDK 10.1.0 + Triton Windows 3.8.0.post29 |
| Общие параметры | context/KV 65536, page 1, BF16 KV, FP32 GDN, prefill 512, MoE 2048/LRU, serial expert load, concurrency 1, graphs 0 |
| Compiler workaround нового стека | Обязателен; сохранённый SHA256 `c30250abfaa0ee14e744e84628458d0f423d82f63759525c5be0591fbd63b19d` |

Версии и hashes взяты из прежних отчётов и manifests, заново не проверялись.
Данные полной серии относятся к `f755ec7c`, сравнение стеков/affinity — к
`3b8bdecb`, новая диагностика подготовлена поверх `dac3d655` с локальными
изменениями. Один HEAD не описывает все эти исполнения.

## 2. Карта выполненных запусков

| Сохранённый запуск | Объём и результат | Stop и границы вывода |
|---|---|---|
| Полная серия, 17:51:58–19:08:25 | 353 корректных запроса; два прохода длинных входов, граница 65280 + 256, 6 пар очереди/отмены, 11 циклов нагрузки | Чистый Stop; стабильность скорости после soak не принята |
| `probe-new`, 20:05:44–20:13:30 | 36 коротких запросов успешны; следующий длинный `UID=37` не завершён | `clean_stop=false`; позднее отдельно зафиксировано освобождение ресурсов |
| `short-rocm1000` и `short-rocm1010b` | 36 + 48 корректных коротких запросов; парное сравнение стеков и два переключения affinity | Оба Stop чистые; affinity на длинном prefill и soak здесь не проверялась |
| `short-one-per-coreb`, 22:02:09–22:05:57 | Readiness отдельно; 1 counting-прогрев + 3 измерения, все 4 ответа правильны | Чистый Stop; короткая приёмка launcher с affinity, без новой progress-диагностики |
| `backend-progress-20261013` | Сохранены 118 успешных CPU-проверок, включая 22 новых случая | Модель и GPU compute не запускались; модельная приёмка открыта |
| Целевой прогон 15.10 | Отложен до Start; short/long/short не выполнялись | Нет новых модельных результатов или нового Stop |

Источники: [полная серия][full-execution], [неудачный probe][probe-execution],
[парное сравнение][comparison], [короткая приёмка affinity][affinity-execution],
[offline qualification новой диагностики][progress-qualification].

## 3. Неудачный длинный запрос: хронология

Ожидалось завершение `decode-63488`, затем короткие контрольные запросы.
Фактически получен HTTP 200 и SSE UID 37, но первого выходного токена,
usage, terminal event и `[DONE]` нет. Дальнейшие запросы не выполнялись.

| Время / показатель | Что сохранилось |
|---|---|
| 20:10:53,627 | Последняя backend telemetry относится к предыдущему короткому запросу, UID 36 |
| 20:12:14 | Последние строки `Prefill batch` для длинного запроса; клиентский счётчик достиг 54 272 |
| Около 20:13:00 | После 45 секунд без нового prefill log/output клиент прекратил запрос; frontend записал `Aborting request for user 37` |
| 20:12:59,592 → 20:13:29,454 | Разрыв между соседними heartbeat наблюдателя **29,863 с**; до него и после него присутствуют записи памяти |
| 20:13:29,481 | Первый сохранённый fault: `System probe 'processes' timed out after 29623ms (limit 5000ms)` |
| 20:13:30 | Frontend сообщил о выходе `freetoken-detokenizer-0`; это позже остановки прогресса и fault |
| 20:13:30,631 | Wrapper завершился с `client_exit=1`, `clean_stop=false`; его отдельная Stop-ошибка — process probe timeout 29366 ms |
| 20:15:05,054 | Сохранён более поздний `controller stop` intent; эта запись не подтверждает успешный Stop предыдущей попытки |
| 20:21:49,935 | Отдельная сохранённая запись восстановления: известных PID и listeners нет, RAM 49,508 / commit 48,273 / GPU free 14,762 |

Источники: [результат UID 37][long-result], [stdout сервера][probe-stdout],
[backend telemetry][probe-telemetry], [observer JSONL][probe-observer],
[fault marker][probe-fault], [Stop intent][probe-stop-intent],
[восстановление ресурсов][recovery].

### Что означают эти записи

- `54 272` — сумма `#new-token` из строк prefill, которую хранит клиентское
  поле `completed_prefill_tokens`. Это последняя наблюдаемая граница прогресса;
  она не указывает последнюю фактически завершённую инструкцию GPU.
- HTTP 200 подтверждает начало HTTP/SSE обмена. Успешного результата генерации
  в данном запросе нет. TTFT и decode TPS для него отсутствуют.
- `BrokenPipeError(10058)` записан после того, как watchdog клиента закрывает
  сокет. Эта запись сама по себе не устанавливает причину остановки backend.
- Process probe timeout — подтверждённый сбой наблюдения/cleanup. Он появился
  после потери прогресса и не объясняет, почему остановился длинный запрос.
- Последний backend host stage и Python stack **не записаны**. Telemetry
  UID 36 не описывает место остановки UID 37. Новая progress-диагностика
  добавлена позднее и не может восстановить этот отсутствующий стек.
- В сохранённом `service-state.json` осталось `RUNNING`, без `stopped_at`.
  Это устаревшая запись состояния после неуспешного Stop, а не доказательство
  работающего процесса после восстановления. Fault marker сохранён.

В выгрузке Windows за 20:05–20:19 не найдено записей сбоя драйвера;
WMI Activity Operational был выключен. Пользователь не находился за ПК и не
подтвердил локальное зависание в этот момент. Отсутствие событий не исключает
сбой GPU/драйвера/ОС. Записан `TdrDelay=8`; настройки не менялись.
Источник: [сохранённые события Windows][windows-events].

## 4. Память

Все числа ниже — **минимальный доступный запас**, а не объём занятой памяти.
Минимумы разных столбцов могли наблюдаться в разное время.

| Запуск | Samples | RAM free | Commit free | GPU local free |
|---|---:|---:|---:|---:|
| Полная серия 353 запросов | 1472 | 23,519 | 10,695 | 6,598 |
| Неудачный длинный probe | 137 | 24,024 | 13,920 | 6,572 |
| Короткое сравнение, ROCm 10.0 | См. исходный manifest | 24,260 | 13,493 | 6,607 |
| Короткое сравнение, ROCm 10.1 | См. исходный manifest | 24,038 | 13,386 | 6,591 |
| Короткая приёмка `one_per_core` | 69 | 21,206 | 8,983 | 6,575 |

Источники: [полная серия][full-report], [сырые samples неудачного probe][probe-observer],
[A/B][comparison], [manifest приёмки affinity][affinity-manifest].

У неудачного probe последняя полная resource sample — 20:12:57,050:
RAM 24,422 / commit 13,949 / GPU free 6,572 GiB. Записанные минимумы выше
рабочих порогов 12/4/3 GiB. **Прямого подтверждения исчерпания памяти нет.**
Измерения не непрерывны; разрыв heartbeat почти на 30 секунд ограничивает
вывод о происходившем между samples.

В полной серии сумма private bytes пяти записанных процессов перед/после soak
снизилась с 7 631 683 584 до 7 261 274 112 байт (**−0,34497 GiB**).
Первоначальное aggregate-поле было null и не используется как доказательство.
Этот пересчёт не показывает рост private bytes между двумя точками, но не
исключает временные пики, другие виды памяти или утечку в другом исполнении.
Источник: [исправленный расчёт private bytes][soak-memory].

Постоянные пороги: startup 38/45/14, runtime 12/4/3, recovery 38/28/14 GiB
в порядке RAM/commit/GPU free. Прежнее разовое снижение startup commit в
копии launcher относилось только к короткой приёмке affinity; после Stop
порог копии возвращён к 45 GiB. На будущий длинный прогон оно не распространяется.

## 5. Скорость: что сравнимо

### Внутри полной серии ROCm 10.1

Контроль — одинаковый короткий запрос 24/391 токен, по три повтора до/после.
Все строки ниже относятся к серверу с ёмкостью 64K.

| Блок | До, токен/с | После, токен/с |
|---|---:|---:|
| Длинная генерация | 50,73 | 50,03 |
| Очередь / отмены | 49,92 | 49,92 |
| Длительная смешанная нагрузка | 50,02 | **39,06** |

После soak снижение **21,91%**, `performance_qualified=false` в сохранённой
приёмке. Два успешных длинных запроса с входом 63 488 имели медиану
TTFT 95,118 с и decode 42,796 токен/с, ответ по 391 токену.
Источник: [полный отчёт JSON][full-report].

### Библиотеки и размещение CPU

В последовательном сравнении стеков после одинаковых 32 коротких прогревов:
ROCm 10.0 — **37,196**, ROCm 10.1 — **36,175 токен/с**. По три контрольных
повтора, разница −2,75%. Ускорение от обновления библиотек не подтверждено;
такой объём и порядок A/B не доказывают устойчивый регресс SDK.

В одном процессе нового стека изменение только affinity backend дало:

| Переключение | Один LP на ядро, `0x5555` | Возврат `0xFFFF` | Прирост относительно соседнего возврата |
|---|---:|---:|---:|
| 1 | 46,740 | 36,490 | +28,09% |
| 2 | 46,307 | 36,977 | +25,23% |

Каждая величина — медиана трёх одинаковых запросов. CPU подготовка шага
уменьшилась примерно с 26,8–26,9 до 20,5–20,6 мс; MoE miss rate в четырёх
блоках одинаковый, 0,149467178. GPU stream metric не уменьшился.
**Эффект affinity воспроизведён.** Конкуренция потоков на SMT — вероятное
объяснение, но прежняя миграция конкретных потоков не записывалась.
Это не доказательство исправления длинного stall или полного объяснения soak.
Источник: [сохранённое A/B и affinity сравнение][comparison].

Короткая приёмка repository launcher с affinity позднее дала **45,882 токен/с**,
TTFT 0,750 с. Она имела один прогрев и другой запуск, поэтому не является
новым парным сравнением SDK. Источник: [summary четырёх counting-запросов][affinity-requests].

Исторический capacity-срез 12 октября на ROCm 10.0: **4K 37,20 / 8K 37,42 /
64K 37,56 токен/с**, по два измерения после одного прогрева. Эти данные
сравнивают ёмкость context/KV при коротком входе, не производительность нового
стека после длинного prefix. Источник: [отчёт этапа 8][capacity-report].

## 6. Stop и ошибки инструментария

- Полная серия: сохранён `STOPPED`, root/observer ownership=false,
  process tree=0, listeners=0, recovery_required=false.
  Источник: [status после Stop][full-stop].
- Успешное A/B: оба Stop чистые; сохранённая сверка не нашла оставшихся
  процессов среди 31 записанной идентичности двух запусков. Источник: [A/B][comparison].
- Приёмка affinity: Stop чистый; в сохранённом результате сверки 23 PID/time
  нет оставшихся процессов. Источник: [manifest][affinity-manifest].
- Длинный probe: **managed Stop не засчитан**. Позднейшее отсутствие PID/порта
  и восстановление ресурсов записаны отдельно, с `managed_stop_succeeded=false`
  и `fault_preserved=true`. Источник: [recovery evidence][recovery].

Отдельно сохранены ошибки harness, которые нельзя считать модельным stall:

1. `short-rocm1010`: унаследованный флаг профилирования создал пустой cProfile,
   который отверг `pstats.Stats`; успешное сравнение находится в свежем `short-rocm1010b`.
2. Первая приёмка affinity: readiness и размещение CPU прошли, counting-клиент
   не передал `stats_poll_s`; Stop чистый, затем выполнена исправленная короткая попытка.
3. Null aggregate private bytes в full-series wrapper: заменён отдельным
   расчётом из сохранённых индивидуальных значений, исходная запись сохранена.

Источники: [отчёт исследования скорости][speed-report],
[отчёт внедрения affinity][affinity-report], [расчёт памяти][soak-memory].

## 7. Что остаётся неизвестным и какой evidence нужен

| Вопрос | Что есть в сохранённых данных | Чего нет |
|---|---|---|
| Почему остановился длинный probe? | UID 37, prefill 54 272, таймлайн, ресурсы, provider timeout | Стек и host stage backend в момент stall; native HIP/GPU trace |
| Почему был разрыв наблюдателя на 29,863 с? | Разрыв между heartbeat и затянувшийся process probe | Источник задержки: ОС, provider, driver или иной фактор |
| Что вызвало снижение после soak? | Контроль до/после; обратимый эффект affinity в другом исполнении | Размещение отдельных потоков во время самого снижения |
| Работает ли новая диагностика с моделью? | 118 CPU-проверок и offline qualification | Целевой модельный short → long → short и Stop |

Раннее полное зависание ПК, упомянутое в истории миграции, — отдельный эпизод.
Совпадение его причины с probe 13 октября не установлено. Исторический checkpoint
2 октября также содержит отдельный NaN/Inf-сбой; его состояние не описывает
позднейший принятый ROCm 10.1 профиль. Источник: [исторический checkpoint][historical-checkpoint].

Следующий согласованный модельный шаг остаётся одним целевым прогоном с
`ProgressDiagnostics` и Stop, когда пользователь возобновит работу.
Нужны `backend-progress/latest.json`, `stacks.log`, request/ack и snapshot
`diagnostic-captures/<id>/capture.json`, связанные с PID и временем создания.
Python stack покажет host frame, но не последнюю исполненную GPU instruction.
При первом fault попытка заканчивается; новая полная серия в этот шаг не входит.
Описание механизма: [progress diagnostics][progress-doc].

## Источники

Ссылки ведут к существующим локальным артефактам. Исходные логи, fault markers,
manifests и прежние неуспешные результаты при подготовке сводки не изменялись.

[full-execution]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/full-series-20261013-1749/execution.json
[full-report]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/full-series-20261013-1749/report.json
[full-stop]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/full-series-20261013-1749/status-after-stop.json
[soak-memory]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/full-series-20261013-1749/soak-memory-growth-validated.json
[probe-execution]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/speed-investigation-20261013/probe-new/execution.json
[long-result]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/speed-investigation-20261013/probe-new/requests/long-63488/result.json
[probe-stdout]: C:/Users/Iskandar/FreeTokenROCm/.local/speed1010-new/runs/ctx64k-20261013-200550-c151cd75/server-logs/stdout.log
[probe-observer]: C:/Users/Iskandar/FreeTokenROCm/.local/speed1010-new/runs/ctx64k-20261013-200550-c151cd75/observer.jsonl
[probe-telemetry]: C:/Users/Iskandar/FreeTokenROCm/.local/speed1010-new/runs/ctx64k-20261013-200550-c151cd75/backend-telemetry.jsonl
[probe-fault]: C:/Users/Iskandar/FreeTokenROCm/.local/speed1010-new/recovery-required.json
[probe-stop-intent]: C:/Users/Iskandar/FreeTokenROCm/.local/speed1010-new/runs/ctx64k-20261013-200550-c151cd75/stop-intent.json
[recovery]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/speed-investigation-20261013/resource-recovery-observed.json
[windows-events]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/speed-investigation-20261013/windows-incident-evidence.json
[comparison]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/speed-investigation-20261013/comparison-verified-final.json
[affinity-execution]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/cpu-affinity-launcher-20261013/short-one-per-coreb/execution.json
[affinity-requests]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/cpu-affinity-launcher-20261013/short-one-per-coreb/requests/summary.json
[affinity-manifest]: C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/constraints/windows-rocm1010-cpu-affinity-launcher-20261013.json
[progress-qualification]: C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/backend-progress-20261013/qualification.json
[speed-report]: C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/rocm1010-speed-investigation-20261013.md
[affinity-report]: C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/rocm1010-cpu-affinity-launcher-20261013.md
[capacity-report]: C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/integration-research-stage8-20261012.md
[historical-checkpoint]: C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/64k-migration-status.md
[progress-doc]: C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/backend-progress-diagnostics-20261013.md
