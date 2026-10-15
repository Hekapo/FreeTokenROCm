# Диагностика остановки прогресса backend — 13.10.2026

Подготовлено в ветке `integration/backend-progress-diagnostics` от `main`
`dac3d655`. Этап выполнен без модели и GPU compute; изменения не закоммичены.

## Причина изменения

В прежнем длинном probe прогресс остановился на54272prefill tokens, затем
системная проверка процессов задержалась и observer начал cleanup.
Причина stall остаётся неизвестной. Старый fault marker и данные сохранены.
Перед следующей попыткой нужен стек backend и описание последнего этапа,
которые не зависят от доступности системного provider.

## Реализация

- Явный `Start -ProgressDiagnostics`; обычный запуск сохраняет режим off.
  Флаг не меняет локальный config/schema3, CPU affinity или numerical settings.
- Bootstrap сначала публикует выбранный GPU, затем импортирует scheduler,
  сохраняя исходный порядок до CUDA init. Scoped hooks устанавливаются только
  в single-rank backend.
  Функции и их аргументы проверяются до изменения; при выходе методы класса
  восстанавливаются, включая наследуемые IO методы.
- Записываются этапы scheduler loop, получение/обработка сообщения,
  планирование batch, восстановление linear state, host forward submission,
  ожидание GPU result, отправка reply, idle integrity и cache rebuild.
  В журнале32 последних события, request UID и cached/device lengths.
  Ни одного дополнительного чтения tensor values или GPU synchronize/query
  для диагностики нет. Host return из forward не объявляется GPU completion.
- CPython watchdog дампит все Python threads после10s непрерывной активной
  операции. При завершении busy period таймер отменяется; обычное ожидание
  входящего сообщения не запускает stall cleanup. После45s активного этапа
  без изменения sequence observer сохраняет данные и выполняет cleanup.
- Перед Stop есть fresh stack request с уникальным ID и проверкой PID/time.
  Ответ ожидается максимум1500ms. При observer fault ожидания нет: critical
  memory/provider cleanup сохраняет прежний порядок остановки.
- Capture обращается к native process identity и локальным файлам, без WMI.
  Ошибка/отсутствие ответа записывается явно и не запрещает cleanup. JSON
  ограничен64KiB, stack snapshot1MiB; полное имя и offsets/truncation сохранены.
  Первый fault и прежняя recovery/quarantine история не перезаписываются.

Артефакты в каждом diagnostic run:

| Файл | Содержание |
|---|---|
| `backend-progress/latest.json` | Текущий этап, identity и последние события |
| `backend-progress/stacks.log` | Watchdog и запрошенные Python thread stacks |
| `backend-progress/capture-request.json`, `capture-ack.json` | ID/identity обмена |
| `diagnostic-captures/<id>/` | Снимки latest/stacks и итог `capture.json` до Stop |

Команды приведены в [руководстве](run-windows-rocm64k.md#progress-diagnostics).
Поведение faulthandler сверено с Context7 и
[документацией CPython3.12](https://docs.python.org/3.12/library/faulthandler.html#dumping-the-tracebacks-after-a-timeout):
используется C watchdog с открытым descriptor, без недоступного в Windows
`register`, без новых аргументов CPython3.15 и без `exit=True`.

## Проверка

До реализации новый набор дал8errors/5failures из-за отсутствующих helper/API;
отчёт `before.xml` сохранён. Итог: **118passed**, включая22 новых случая.
Общий GPU teardown fixture отключён через `--confcutdir=tests/utils`;
команды запускались с `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Проверены:

- idle/busy transitions, вложенные этапы и отмена таймера;
- сохранение исходной ошибки worker и ограниченный журнал;
- подписи hooks против AST текущего source, возврат результата и восстановление
  класса; повторное наложение hooks отвергается;
- публикация GPU до scheduler import/hooks; новый regression test сначала
  выявил нарушение порядка, затем прошёл после исправления;
- неправильный PID, повторно использованный PID/time, неверный/старый ack,
  повреждённый и слишком большой request;
- bounded capture без системного provider; timeout сохраняет уже имеющийся
  dump и не притворяется успешным fresh capture;
- реальный CPython3.12 Windows процесс: `ctypes.PyDLL('kernel32').Sleep`
  удерживал GIL, а C watchdog всё равно записал ожидаемый Python frame;
- живой CPU worker: PowerShell проверил его настоящий PID/time, получил ack
  и сохранил stack/journal до завершения именно этого owned процесса;
- монотонный elapsed-time detector: здоровый прогресс, простой и новая sequence
  не вызывают ложный stall.

`Verify` прошёл для pinned ROCm10.1 и ROCm10.0. Runtime lists теперь22/20files;
source476, package74 и native5 проверяются прежними gates. Compiler workaround
`c30250ab...` не изменён. Постоянные startup38/45/14, runtime12/4/3,
recovery38/28/14GiB сохранены. `Status` — STOPPED, процессов/listener0,
recovery_required=false. Модель для этих проверок не запускалась.

Release records связывают новую упаковку с предшествующим launcher record;
новый model smoke и `progress_diagnostics_model_qualified` отмечены false.
Предыдущая успешная CPU-affinity smoke записана как historical inheritance,
а не новая приёмка этого bootstrap.

Первый118-case batch имел117passes и один отказ hash gate после финальной
правки helper до обновления release hash. Этот XML сохранён отдельно;
runtime records обновлены, итоговый batch проверяет согласованные hashes.

## Область результата и следующий шаг

Готов механизм сбора evidence, причина прежнего stall ещё не установлена.
Python traceback показывает host frame, но не native HIP stack или последнюю
фактически исполненную GPU instruction. Если Python-thread reply недоступен,
есть независимый C watchdog; при остановке раньше его deadline fresh stack
может отсутствовать, что явно отражается в capture metadata. Зависание всей
ОС или диска не даёт гарантии сохранения файла.

Следующая приёмка — один целевой diagnostic short → прежний long → short
с новым recorder и Stop, с действующим memory admission. Новая полная серия
и soak не требуются. После первого fault попытка заканчивается; сохранённые
stacks/latest определяют следующий конкретный шаг, вместо повторения без
новой гипотезы. Diagnostic timing не используется для вывода о скорости.

[Машинный manifest](../constraints/windows-rocm64k-backend-progress-20261013.json)
фиксирует CPU results, hashes runtime и проверочные записи. Raw audit:
`rocm10-20260926/research-refresh-20261011-audit/deps-upgrade-20261012/backend-progress-20261013`.

## Сводка сохранённых логов — 15.10.2026

[Сводный отчёт](C:/Users/Iskandar/FreeTokenROCm/FreeTokenROCm/repos/wt-triton38-align-20260928/docs/64k-saved-logs-summary-20261015.md)
объединяет полную серию, остановившийся длинный probe, сравнение SDK/affinity
и короткую приёмку launcher. В нём приведены таймлайн, свободная память,
результаты Stop и ссылки на первичные данные. Целевой прогон с новой
диагностикой 15 октября отложен до Start по просьбе пользователя.
Сводка составлена без новых запусков модели и проверок;
`progress_diagnostics_model_qualified` остаётся false.
