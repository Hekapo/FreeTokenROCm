# Этап 1: диагностика non-finite и границы GDN — офлайн-часть

Дата: 02.10.2026. Ветка: `integration/research-refresh-20261002`. Точка начала: `e5e2899eef36e55eb84a58a4cd59de8766ac8c46` (план поверх сохранённой миграции `c4832f35055c5f27998fd36996d21d4dbda18b76`). Изменения этого этапа пока находятся в рабочем дереве.

**Статус: офлайн-часть выполнена; модельная приёмка этапа 1 не выполнена. Причина прежнего NaN на 64K не локализована.** Пользователь сообщил, что сейчас не может освободить память. Модель и GPU kernels в этой попытке не запускались. Старые managed runtime files, quarantine/recovery markers, packages, SDK, driver и пороги памяти не изменялись.

## 1A. Диагностика первой нарушенной границы

Добавлен `python/freetoken/diagnostics.py`, подключённый к Qwen3.5/3.6 MoE и используемому им chunk GDN. По умолчанию монитор отсутствует: нет finite-check, копий состояния, device scalar reads и файлов снимков. Диагностика включается переменной `FT_FINITE_DIAGNOSTICS_DIR` **до импорта модели** и проверяет только prefill.

Проверяемые границы:

- embedding, нормы/residual, mixer, MLP, итоговая норма и LM head;
- GDN projection, используемое входное conv state, convolution, gates, L2 norm;
- chunk inputs, cumsum, KKT/solve, WY recompute, используемое recurrent state до/после обновления, chunk output, gated norm и out projection;
- MoE router, shared expert, shared gate, routed experts и их сумма.

На первом NaN/Inf создаются `record.json` и `tensors.pt`, затем выбрасывается `FirstNonFiniteError`. Следующий forward того же монитора запрещён. Запись содержит phase, номер forward, layer, query/cached lengths, shape/dtype/stride, индекс первого плохого элемента, использованные state slots и локальный контракт `[slot,H,V,K]`. Для GDN операций сохраняются tile/config/autotune metadata. Для token outputs вычисляются request, chunk внутри query и абсолютная позиция с учётом cached prefix. Для `h` отмечено, что snapshot соответствует **началу** chunk; для конечного state slot точный момент внутри kernel неизвестен.

Суммарный размер двух файлов ограничен 64 MiB с резервом для служебных данных. Сохраняются сбойный output и непосредственные inputs; большие тензоры обрезаются до ограниченного префикса, отдельный первый плохой scalar сохраняется всегда при успешной записи. `detach().cpu().clone()` не сохраняет backing storage всего pool через маленький view. Для in-place state update предварительно сохраняются значения выбранных слотов. Неиспользуемые/неинициализированные слоты не проверяются. Ошибка записи не скрывает первоначальный non-finite.

`FT_FINITE_START_FORWARD` и `FT_FINITE_END_FORWARD` задают необязательный включительный интервал, начиная с 1. Счётчик относится к вызовам model forward в новом процессе, включая warmup; его нельзя автоматически приравнивать к историческому engine counter 253. Проверка и CPU snapshots синхронизируют GPU и меняют timing. Этот режим требует `cuda_graph_bs=0`; его скорость не является обычной скоростью модели.

Для будущего managed запуска:

1. Заморозить отдельный candidate source, проверить imported paths и manifests, сохранить старый stack/settings и отдельные cache roots.
2. Передать абсолютный `FT_FINITE_DIAGNOSTICS_DIR` внутри нового run directory через существующий managed launcher. Он наследует environment; новые переменные не входят в его список очистки. Проверку наследования в реальном backend ещё предстоит выполнить.
3. Использовать существующую процедуру diagnostic/recovery admission, startup gate 45 GiB свободного commit, прежний observer и managed Stop. Не удалять recovery/quarantine вручную.
4. Один bounded запуск: readiness, короткий запрос, один прежний длинный ввод. Первый non-finite завершает попытку. Затем выключить диагностику; обычный long smoke проверяется отдельно.

## 1B. Подтверждённые нарушения математического контракта

В шести модулях FLA (`l2norm`, `cumsum`, `chunk_fwd`, `wy_fast`, `chunk_delta_h`, `chunk_o`) 39 block-pointer loads имели `boundary_check`, но не задавали padding. У используемого Triton 3.8 `padding_option=""` допускает неопределённое значение, включая NaN. В dot/reduction нельзя полагаться на последующий mask: `0 * NaN` остаётся NaN. Это согласуется с masked-load контрактом [Triton](https://triton-lang.org/main/python-api/generated/triton.language.load.html); точная block-pointer семантика проверена также в `triton/language/core.py` установленной версии 3.8.0.post28.

Добавлен явный `padding_option="zero"` до dot/exp/reduction. Это локальный перенос инварианта из [FLA #1062, revision cfab1960f257bd083e3da00cceb66dd64c885692](https://github.com/fla-org/flash-linear-attention/commit/cfab1960f257bd083e3da00cceb66dd64c885692), без замены всей реализации на upstream explicit pointers. Сохраняются существующие GQA, indexed state `[V,K]`, dtype/precision и настройки kernels; copyright/attribution vendored FLA сохранены.

Также расширены base offsets до int64 **перед** умножениями. Особенно важны `boh * H * V * K` и `state_slot * H * V * K`: преобразование уже переполненного int32 не исправляет адрес. Это защитный контракт для больших offsets, а не установленная причина прежнего сбоя при текущем размере pool.

### Численная проверка

`tests/kernels/test_fla_padding.py` запускает реальные JIT kernel bodies в отдельном CPU interpreter process. Неопределённые masked float loads намеренно возвращают NaN; явные zero loads сохраняют ноль. Обычный интерпретатор подставляет нули и скрывал бы нарушение.

До исправления fixture завершился `AssertionError: l2norm padding`. После исправления прошла цепочка L2 norm → cumsum → KKT/solve → WY → indexed state update → output с независимой последовательной CPU-double recurrence:

- lengths `1,15,16,17,63,64,65`, H/Hg=2/1, K/V=17/19;
- lengths `17,65`, геометрия модели H/Hg=32/16, K/V=128/128;
- непустой initial state, reordered slots, varlen/nonzero sequence offsets;
- NaN в невыбранном pool slot остаётся нетронутым;
- FP16 inputs, FP32 gates/state, CPU-double reference; заранее заданные `rtol=0.008`, `atol=0.003` для output и final state;
- адреса реального state kernel на offsets >=2^31 элементов проверены перехватом load/store адресов без их разыменования и без гигантских allocations.

В интерпретаторе обходится GPU autotuner (фиксированный BK=32 только в fixture), `tl.exp` aliases разрешаются в момент вызова, а cumsum запускается напрямую без GPU context wrapper. Production autotune не меняется. Fixture подтверждает arithmetic/masks/indexing; он не подтверждает BF16 rounding, GPU compilation, LDS либо поведение driver.

**Связь с прежним отказом не установлена.** Исторический отказ после 4096 long-prefill tokens произошёл на forward с query chunk 512, кратным 64; K/V=128 также полностью покрывают tiles. Приведённый reproducer доказывает нарушение partial-tile контракта, но не объясняет тот конкретный NaN.

## 1C. LDS: результат сравнения, без нового tuning

Прочитан [донор FLA #751, revision 85856e259f6184bb5f16dcf686b2fea41b6b7d34](https://github.com/fla-org/flash-linear-attention/commit/85856e259f6184bb5f16dcf686b2fea41b6b7d34). Он ограничивает upstream BV/pipeline stages и autotune candidates; backward изменения к нашей inference-only версии не применяются. В локальном state kernel уже одна конфигурация BV=32, num_warps=4, num_stages=2 — она также исключает повторное повреждение in-place state при autotune.

Локальный `chunk_o` использует фиксированные BK=128/BV=64, а `BKV_LIST` относится к закомментированному autotune и не выбирает реальный launch. Простое изменение этого списка не исправило бы активный путь. Новый pruning или tile selection не перенесён без подтверждения metadata компилятора именно изменённого source. Историческая численная проверка всех шести KKT settings прошла; BK=32 не закреплён как предполагаемое лекарство. **LDS новой GPU compilation остаётся отдельной непроверенной частью.**

## 1D. Native Q4_1 и ownership: аудит source

Прослежен путь `gguf_expert_specs/load_gguf_expert_sources` → persistent host banks → `OffloadMoeCache` → `fused_experts_gguf` → native GGUF kernels:

- Q4_1: 32 элемента / 20 байт блока. Expert-major `[E,2I,row_bytes(H)]` и `[E,H,row_bytes(I)]` имеют собственный точный stride. Добавлять padding внутрь bank rows нельзя: MMVQ рассчитывает плотный адрес `expert*nrows*blocks_per_row`.
- GPU cache выделяется `torch.zeros`; copy переносит полные expert rows. В MMVQ есть row bound и `i < blocks_per_row`. Q4_1 vecdot читает quants внутри блока; affine `m4*s8` сохранён. В quantized activation выделяется физическая область до кратности 512, и quantizer явно записывает нули за исходным input dim.
- MMQ loaders проверяют/clamp последние rows; tile-wide K reads требуют кратности собственной tile ширине. В эту попытку native kernels не менялись; произвольные новые shapes и poisoned physical tails аппаратно не проверялись.
- Host banks удерживаются `bank_sources` и process-lifetime mmap ownership. Основной copy и compute используют один stream. Для overlap существуют begin fence, ready/release events и ожидание release перед повторным использованием буфера; pageable whole-layer branch выполняет обычный blocking copy. Не найден конкретный ранний free/reuse в просмотренном пути.

Это source audit, а не доказательство отсутствия driver/copy faults. [RDNA r31](https://github.com/stew675/llama-cpp-rdna-boosts/releases/tag/v16-84e76d8a2-r31) и [ROCm #12535](https://github.com/ROCm/rocm-systems/issues/12535) использованы как гипотезы из плана; их kernels/SDMA workaround не переносились.

## Выполненные checks и следующий шаг

CPU-only проверка на текущем Python 3.12 / Torch 2.13.0+rocm10.0.0 / Triton Windows 3.8.0.post28:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'python'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HIP_VISIBLE_DEVICES = '-1'
$env:ROCR_VISIBLE_DEVICES = '-1'
$env:CUDA_VISIBLE_DEVICES = '-1'
python -B -c 'import torch, pytest, sys; sys.exit(pytest.main(["-q","-o","addopts=","-p","no:cacheprovider","tests/utils/test_finite_diagnostics.py","tests/kernels/test_fla_padding.py","tests/models/test_qwen35_gdn_prefill.py"]))'
```

**15 passed, 2 skipped, 1 warning, 17.18 s.** Пропущены две GPU convolution проверки. Предупреждение — ожидаемый FLA CPU fallback. Новые проверки покрывают выключенную диагностику, первый failure/запрет продолжения, размер и storage snapshots, in-place state values, реальную границу chunk pipeline, MoE router, scope LM head и varlen location. `git diff --check` прошёл.

Далее: при доступной памяти подготовить один managed diagnostic candidate и проверить GPU compilation/LDS и первую non-finite границу. При её обнаружении чинить узкий подтверждённый контракт. Затем отдельный ordinary long smoke. До этого этап 1 не закрыт, этап 2 не начинается и новых показателей скорости нет.
