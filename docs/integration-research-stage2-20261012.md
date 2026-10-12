# Интеграция исследования: этап 2, 12.10.2026

**Статус: завершён для текущего профиля** Windows / RX 9070 XT / gfx1201, Qwen3.6-35B-A3B GGUF Q4_1, context/KV 65536, BF16 KV, FP32 GDN state, prefill512, MoE2048/LRU, concurrency1, graphs0, serial expert load. CPU/address, GPU operators, диагностический и обычный long smoke пройдены; оба managed Stop чистые. GPU gate сэмплера этапа1 также пройден.

Ветка `integration/research-refresh-20261002`, HEAD `e5e2899eef36e55eb84a58a4cd59de8766ac8c46`; source остаётся незакоммиченным. Packages/driver не обновлялись: Torch2.13.0+rocm10.0.0, SDK10.0.0 (Torch сообщает HIP7.15.26333), Triton Windows3.8.0.post28, TVM-FFI0.1.13.post3. Сохраняется прежний launcher с process-start caching `use_buffer_ops=true` и импортом Torch до Triton; перенос knob в production относится к отдельному этапу6.

## 2.1. FLA offsets

[FLA1329](https://github.com/fla-org/flash-linear-attention/pull/1329), merged `b4a0c7344078f28cd5acfee9533feba8c9213dae`, повторно проверен перед переносом.

- В `kernel/fla/fused_sigmoid_gating_recurrent.py` program IDs, state-slot/cache indices, parent indices и step index расширены до int64 до multiplication chains.
- Новый случай в существующем `tests/kernels/test_fla_padding.py` исполняет настоящую recurrent JIT arithmetic через CPU interpreter. Float-pointer loads/stores перехватываются: проверяются виртуальные адреса >2^31, память большого pool не выделяется. До правки воспроизведён `recurrent state read offset overflow`, после проходит initial/final state и intermediate/tree-cache addressing.
- Прежние zero-padding исправления сохранены. Numerical CPU chunk reference и проверка undefined loads продолжают проходить.
- Recurrent grid текущего профиля: `(1,4,N*32)` при N<=1. Лимит y/z не достигается; grid и floating-point formula не менялись. Большие packed batches остаются отдельным контрактом, эта приёмка их не утверждает.
- GPU fixture использует BF16 inputs, FP32 pool, H/Hg/K/V=32/16/128/128, T=65 и непустой state-slot2; сравнивает partial chunk и recurrent decode с независимой CPU-double recurrence, проверяет неизменность трёх невыбранных NaN slots. Допуски заданы до выполнения: rtol0.025/atol0.001.

Фактический LDS: state36864 bytes, output16384, WY16384, KKT4096, recurrent128. State config BV32/warps4/stages2 и output BK128/BV64/warps4/stages2 сохранены.

## 2.2. GGUF launch contract

[FreeToken185](https://github.com/FlashML-org/FreeToken/pull/185), open head `3ca0ccd0365cfe803c959221b7ea442e7167c8f2`, автор erichstuntebeck; перенесены необходимые source hunks с локальным коротким комментарием.

- `hipDeviceGetAttribute` из установленного SDK подтвердил gfx1201 limits: X=2147483647, Y/Z=65535. Enum IDs29/30/31 взяты из установленного `hip_runtime_api.h`; результат в `hip-grid.json`.
- Во всех19launch sites `kernel/csrc/gguf/moe_vec.cuh` flat token/topk перенесён с z на x, rows с x на y. Expert-major Q4_1 packing/stride и affine math сохранены.
- После каждого launch выполняется CUDA/HIP last-error check с grid/block geometry. Это обнаружение configuration error без дополнительной device synchronization; async execution failures остаются ошибками runtime. Семантика проверена по [официальной HIP документации](https://github.com/ROCm/HIP/blob/develop/docs/how-to/hip_runtime_api/error_handling.rst).
- Новый GPU fixture в `tests/kernels/test_gguf_rocm.py`: Q4_1, BF16, 8experts,17rows,32columns; gate/up8192tokens*8routes и down65536tokens*1route. Full outputs побитово равны split batches и совпадают с независимым CPU affine-weight reference (rtol0.02/atol0.1).
- Использован fresh native build/cache. Полная модель использует собственные свежие caches, без перезаписи исходного checkout или прежних binaries.

При prefill512 текущие4096routes не достигали старого grid.z предела. Исправление этого launch contract само по себе не объясняет прежний prefill NaN.

## 2.3. Остаток134/638

- [FreeToken134](https://github.com/FlashML-org/FreeToken/pull/134): flashinfer/sgl_kernel/triton_kernels и CUDA driver probe уже имеют HIP guards. Добавлен недостающий guard `is_vllm_installed`: оба текущих local callers выбирают CUDA-only FP4 Marlin. До правки CPU regression1failed/1passed, после2passed.
- [FreeToken638](https://github.com/FlashML-org/FreeToken/pull/638): local `SymbolicSize` имеет отдельный `m_has_value`, поэтому zero size поддерживается без изменения sentinel. `TensorMatcher` сохраняет symbol options и отдельно проверяет matcher-specific allowed dtype/device lists. HIP PDL path уже реализован локально. Дополнительный перенос этих hunks не потребовался; `tensor.h` не менялся.

## 2.4. Модельная приёмка

Оба запуска ограничены одной последовательностью: short26 → исторический decode63488(max512) → short26. Без request warmup loops и soak. Фиксированные deadlines180s/request,360s/client,45s/no-progress, scheduler stacks10s; startup commit45GiB, runtime RAM/commit/local-VRAM12/4/3GiB. Moonlight отсутствовал. Observer, instance checks, token/semantic validation, KV release и managed Stop действовали.

| Запуск | Finite checks | Short до, total | Long TTFT | Long total | Short после, total | Результат |
|---|---|---|---|---|---|---|
| `ctx64k-20261012-120020-efff1fa5` | On, все prefill forwards с1 | 1.406s | 155.297s | 165.047s | 1.063s | 3/3pass, cleanStop |
| `ctx64k-20261012-120810-a2e405e7` | Off, обычный execution path | 1.125s | 95.172s | 102.656s | 0.797s | 3/3pass, cleanStop |

В каждом long: actual prompt63488, completion391, finish_reason=stop; ответ содержит требуемую последовательность1..100. Все short: expected17, completion3. SHA256 content long и short совпали между режимами; short до/после совпали. After-request KV occupancy0.

Backend inheritance finite-dir/start-forward подтверждено per-PID `stable-triton-knob.jsonl`, включая `__mp_main__`. First-NaN record/snapshot не создавались: все проверяемые границы finite. Snapshot cap64MiB сохранён. Обычный режим не включал finite-directory; прежний guard logits-check64 остаётся частью контрольного launcher.

| Минимумы observer | Диагностический | Обычный | Runtime floor |
|---|---|---|---|
| Free RAM | 24.386GiB | 26.987GiB | 12GiB |
| Free commit | 15.288GiB | 14.895GiB | 4GiB |
| Free local VRAM | 6.943GiB | 6.935GiB | 3GiB |

AfterStop: STOPPED, root_owned=false, observer_owned=false, process_tree_count0, listeners0, recovery_required=false. Во время managed Stop supervisor зарегистрировал выход detokenizer; stop intent уже был выставлен, новый recovery fault не возник.

**Вывод:** прежний воспроизводимый long отказ после интеграции не повторился в диагностическом и обычном режимах. Это квалификация exact candidate и текущего профиля. Историческая первопричина NaN не доказана отдельно: несколько source fixes внесены до сравнения, а timing finite mode отличается. Широкая повторная серия не проводилась.

TTFT/total выше — наблюдаемые timings; diagnostic timings не benchmark. Клиент не записывал per-token trace, поэтому его `decode_tps`/`effective_tps` поля null. Новое сопоставимое сравнение скорости4K/8K/64K остаётся этапом8; время запроса не выдаётся за такой benchmark.

## Recovery и сохранение контроля

До запуска проверен точный recovery fault02.10: non-finite logits, phase=prefill/forward253. Узкий review проверил14 записанных process identities, отсутствие сервера/compiler/listener и восстановление ресурсов. Старые state/logs, fault/stop copies и quarantine сохранены; создан derived STOPPED state. Неуспешный исторический run не переобозначен как успешный.

После перезагрузки GPU DXGI LUID изменился. Перепривязка guard выполнена к единственному DXGI RX9070XT/full-adapter counter; изменён только `gpu_instance`, thresholds сохранены. Свободный commit перед запусками49.390/49.313GiB.

Legacy quarantine остаётся историей; default working-release marker автоматически не выпускался. Для следующего source change аппаратная проверка требует нового manifest. Manifest текущих runs фиксирует дерево до этих documentation updates; production bytes после проверки не менялись.

## Проверки и артефакты

- CPU FLA/diagnostics:15passed/2GPU skipped,17.22s; backend2passed. Virtual-address reproducer и backend guard падали до правок.
- GPU unique contracts:4passed — sampler stream/counts, GDN partial chunk+decode, Q4_1 gate/up и down boundary. GDN/sampler accepted в `s2op-20261012-a4`, GGUF accepted в `s2op-20261012-a5` (79.45s вместе с native build).
- Ранние attempts a1–a4 содержат исправления тестового harness: empty GPU visibility, неверный return unpack в fixture, отсутствие CPU Q4_1 dequant в проекте. Эти fixture/setup failures не объявляются production defects. После GDN/sampler успеха повторялись только GGUF cases.
- PowerShell parser для изменённых controller scripts и AST finite client прошли. `git diff --check` прошёл.

Основной audit: `rocm10-20260926/research-refresh-20261011-audit/stage2-20261012/` от общего workspace. Содержит baseline/control-source, donor metadata, CPU before/after, hip-grid.json, guard/recovery review, control-runtime, candidate-state.json, finite-plan.json/ordinary-plan.json, acceptance.json, verification.json и раздельные source patches.

GPU/model artifacts: `rocm10-20260926/results/s2op-20261012-a4`, `s2op-20261012-a5` и два `ctx64k-*` каталога из таблицы. Каждый model run содержит `finite-smoke/summary.json`, request results, observer.jsonl, startup/native logs, compiled caches и `short-diagnostic-status-after.json`.

Откат одного source пункта по сохранённым bytes/patch, сохраняя отдельные stage1/FLA changes. Runtime-controller snapshots находятся в `control-runtime/`. Не использовать reset hard и не удалять исторический quarantine. Commit/push/install не выполнялись.

Следующий согласованный этап3: отдельные API/tokenizer608, conditional FP32 mRoPE641 и cache lifecycle650 contracts согласно плану. Повторять long smoke следует только при изменении затронутого runtime или конкретном новом риске.
