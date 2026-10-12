# Этап 8: итоговый профиль и сравнение 4K / 8K / 64K

**12.10.2026 — итоговый срез завершён.** Девять коротких запросов прошли, ответы одинаковы, все три managed Stop чистые. Production source на этапе 8 не менялся. Итоговый профиль: принятые source changes этапов 1–4 и opt-in опции 6, qualified Python74, прежний native ROCm10.0 / Torch2.13 / Triton post28. Обе новые опции этапа 6 выключены; ordinary prefill overlap и лабораторный stable knob включены.

## 1. Exact runtime и профиль

Machine: Ryzen7 9700X, RX9070XT / gfx1201 / 16GiB VRAM, visible RAM63,61GiB; Windows11 build26100, DriverStore32.0.31041.1004. Hardware snapshot сохранён в audit.

- Branch `integration/research-refresh-20261002`, HEAD `e5e2899eef36e55eb84a58a4cd59de8766ac8c46` **с незакоммиченными изменениями**, идентифицируемыми frozen manifest. Один HEAD не описывает проверенную версию.
- Runtime `.venvs/freetoken-pydeps-20261012-py312`: CPython3.12.0, 74packages; Transformers5.19.0, Hub1.33.0, FastAPI0.143.0, Pydantic2.14.1/core2.50.1, OTelAPI1.44.0. Native Torch2.13.0+rocm10.0.0 / SDK10.0.0 / Triton Windows3.8.0.post28; TVM-FFI0.1.13.post3, OpenAI2.54.0, NumPy2.4.6, flashlib0.3.0.
- Qwen3.6-35B-A3B GGUF Q4_1; BF16KV / FP32GDN, page1, prefill512, MoE2048/LRU, serial expert load, backend/frontend concurrency1, graphs0, naive cache, memory ratio0.80. Server output default2048, request budget512; greedy temperature0, thinking=false.
- `AMDGCN_USE_BUFFER_OPS=1`, cached once at process start by existing lab helper; Torch импортирован первым во всех recorded processes. `FREETOKEN_TRITON_CACHE_AMD_KNOB=0`, demand threshold0. Per-token trace, profiler, scheduler stacks и finite snapshots выключены; одинаковый logit safety every64 оставлен во всех phases. Moonlight отсутствует.
- Для сравнения лимит контекста **и KV allocation** меняются вместе: 4096 / 8192 / 65536. Остальные frozen inputs одинаковы; 4K здесь не исторический профиль с KV8192/concurrency2.

[Полный profile/package/compiler manifest](../constraints/windows-rocm1000-profile-20261012.json), [dependency constraints](../constraints/windows-rocm1000-pydeps-20261012.txt).

## 2. Сопоставимые короткие замеры

Prompt `Print 1 to 100, comma-separated.` без завершающего newline: **24 rendered tokens**. Один warmup + два measured requests на capacity; warmup исключён из медиан. Все 9 ответов содержат правильные числа1…100, 391completion tokens, reasoning пуст, finish=stop; contentSHA `34a4f1e5bb080915a30b7f67a8546b8e72da130622436caa0fcb81a2eb62c0ee`. После каждого запроса тот же instance и idle KV подтверждены.

| Capacity / KV pages | Correctness | TTFT, s | Decode, tokens/s | Два measured значения, min–max | Total, s | Prompt / output |
|---|---|---:|---:|---:|---:|---:|
| 4K / 4096 | 3/3 | 0.758 | 37.20 | 37.09–37.31 | 11.242 | 24 / 391 |
| 8K / 8192 | 3/3 | 0.758 | 37.42 | 37.31–37.54 | 11.188 | 24 / 391 |
| 64K / 65536 | 3/3 | 0.758 | 37.56 | 37.53–37.59 | 11.141 | 24 / 391 |

64K относительно 4K: **+0.97%**, относительно 8K: **+0.37%** по медиане decode. Это наблюдение на двух measured запросах каждого размера; оно не доказывает улучшение длительной производительности. Historical40.05/40.27/41.40TPS сохранены в плане как reference другой серии; они не controls сегодняшнего опыта. Результаты этапа6.2 имеют другой runtime/profile и не используются для isolated gain.

Метрика: `(reported completion tokens − первый token) / (finish SSE event − первый output SSE event)`. Первый output строго `1`, его single-token encoding проверен отдельно. Finish/EOS tail включён; warmup и loading/JIT исключены. Windows CPython3.12 clock имеет примерно15,6ms granularity. **Prefill throughput не измерен отдельно (N/A)**: TTFT включает scheduler, expert admission и первый decode; деление24/TTFT не является чистым prefill TPS.

| Capacity | Min free RAM, GiB | Min free commit, GiB | Min free local VRAM, GiB | Observer samples |
|---|---:|---:|---:|---:|
| 4K | 25.840 | 14.324 | 8.067 | 60 |
| 8K | 25.554 | 13.961 | 7.990 | 60 |
| 64K | 25.774 | 12.978 | 6.903 | 59 |

Minima относятся ко всему lifecycle, включая loading/JIT/warmup. Startup threshold45GiB free commit и runtime floors12/4/3GiB сохранены. Все after-stop: STOPPED, root/observer/tree/listener0, recovery=false. Backend supervisor exit lines возникают после stop-intent; до него таких failures нет. Historical quarantine/evidence сохранены.

## 3. Учёт принятых и отложенных изменений

| Этап | Решение и границы | Evidence |
|---|---|---|
| 1: FT593 / 599 / 527 | FTW incomplete-output guard, KV page accounting, request-owned penalties приняты; CPU181 и отдельный GPU sampler подтверждены | [Этап1](integration-research-stage1-20261012.md), [sampler в этапе2](integration-research-stage2-20261012.md) |
| 2: FLA1329 / GGUF185 / guard134 | Address/int64, padding/L2 и Q4 boundaries приняты; GPU4, diagnostic3/3 и ordinary3/3 включая63488-token prefill | [Этап2](integration-research-stage2-20261012.md) |
| 3: FT608 / 650 | Tokenizer system folding и live cache rebuild/API приняты; CPU243, model3/3. FT641 неприменим к отсутствующему fallback | [Этап3](integration-research-stage3-20261012.md) |
| 4: Python deps | Qualified74 принят; tokenizer IDs/API parity, CPU259+3, model3/3. ModelScope optional и не установлен | [Этап4](integration-research-stage4-20261012.md) |
| 5: Vendor10.1 / Torch2.14 / post29 | Подготовлен отдельно, **не принят**: histogram и два loop comparisons miscompiled. Native build/CPU298 не заменяют compiler/GPU gates; production bypass отсутствует | [Этап5](integration-research-stage5-20261012.md) |
| 6.1: Portable AMD knob | Opt-in реализован, default off; CPU28/model9passed, дополнительный speed gain не установлен | [Этап6.1](integration-research-stage6-20261012.md) |
| 6.2: Routed short Q4 prefill | Opt-in default0 принят для non-overlap short Q4_1; CPU42/GPU2/model9passed, TTFT−69,81% в отдельном workload. Ordinary overlap не переключён | [Этап6.2](integration-research-stage6b-20261012.md) |
| 7: Prefix reuse | Optional функция отложена; не включена в итоговый профиль | [План](research-integration-plan-20261012.md) |
| 8: Matched capacity + ledger | 9/9short, 3cleanStop, единый профиль и exact manifests сохранены | Этот отчёт |

TVM-FFI0.1.14.post1, Numba/llvmlite, attention632, bounded workspace, GEMV/GDN дополнительные candidates остаются условными/отложенными по [полному учёту](research-integration-plan-20261012.md). Они не выданы за интегрированные изменения.

## 4. Область длинного теста и remaining gates

Ordinary результат этапа2: **63488 prompt / 391 output**, TTFT95.172s, total102.656s, корректный ответ, KV released и cleanStop. Он сохранён как evidence численного long64K пути. 14 numerical source files byte-identical к snapshot после этого теста; native файлы Python candidate совпадают с прежним control. Tokenizer parity, API/cache изменения и default-off ветки проверены на соответствующих этапах.

**Полное итоговое дерево с Python74 на длинном prefix сегодня не прогонялось.** Stage8 подтверждает short capacity comparison; старый long результат не переименован в новый прогон. Повтор широкого long тестирования не выполнен по согласованному сокращению тестов и неизменности numerical/native path. Расширение demand path на overlap/long остаётся отдельной задачей.

Legacy working-release schema1 требует исторические602/62/3 и другой runtime manifest. Текущие9requests в него не подставлены: release marker не promoted, quarantine не снят, general managed Start всё ещё выбирает исходный control. Этот integration/profile record сохраняет текущие результаты и не изменяет те старые gates.

Отдельно открыты compiler qualification ROCm10.1/post29 и его supported OS/driver row; optional prefix reuse и расширение stage6 опций. Merge/publish в этом этапе не выполнялись.

## 5. Изменения harness, воспроизведение и откат

Добавлены strict frozen capacity plan reader и stdlib client. Controller передаёт candidate74/runtime, соответствующий KV budget и новый bounded launch selector. Stable knob раньше допускался только при65536; первый4Kattempt остановлен **до модели**. Теперь4096/8192 разрешены только для managed matched profile с equal KV/capacity, single-request limits, logit64, ResourceGuard/StateManifest/owned pointer, unchanged ordinary prefill; 19 offline guard cases прошли. Existing64K gate, Stop ownership и memory thresholds сохранены.

Для каждой phase отдельные native/JIT/cache roots. Перед Start и после Stop frozen source проверен; lab17files, plan/spec/package SHA256 неизменны во всей серии. Включены CPU-free client simulation и package/native/source verification; повтор всего production CPU набора не потребовался, source на этапе8 не менялся.

```powershell
& C:\Users\Iskandar\FreeTokenROCm\rocm10-20260926\triton38-lab-20260928\run_ctx64k_short_diagnostic.ps1 -Plan <audit>\4k-plan.json -Run
```

Та же команда с8k/64k-plan. Plans привязаны к literal checkout/venv и exact pre-doc source state; последующая документация меняет полную git fingerprint. Для повторного запуска надо сохранить новый reviewable freeze, а не отключать verification. Stop выполняется wrapper в finally, observer и45GiB startup gate обязательны. Порядок текущей серии4K→8K→64K; randomization/long-soak не заявлены.

Audit root: [stage8-20261012](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/stage8-20261012); `source-state.json`, `runtime-manifest.json`, `packages.json`, `hardware.json`, `spec.json`, `runtime-comparison.json`, `pre-documentation-state.json`, `verification.json`, `docs.patch`, `lab.patch`, `STATUS.txt`. Full source fingerprint SHA `2d2f5d7f89c93cfd8c345a04b1422837540c308d0f3252fe501bf4d176597a74`, runtime17SHA `6a197f48cc3fa54d6c889b31e266ca80780f4a06aaf7f188365615b6d2314246`; compiler/DLL hashes в profileJSON.

- 4K: [capacity-4k-20261012-190422-a404f278](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/capacity-4k-20261012-190422-a404f278).
- 8K: [capacity-8k-20261012-190837-882e5f1c](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/capacity-8k-20261012-190837-882e5f1c).
- 64K: [capacity-64k-20261012-191254-b5b06e1f](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/capacity-64k-20261012-191254-b5b06e1f).

Откат этапа8: восстановить только его lab files из `lab-before` / `lab.patch` и docs из `before-docs` / `docs.patch`. Source changes1–6 и прежние evidence не затрагиваются. Для dependency rollback выбрать сохранённый control73/constraints и его отдельные caches. Branch/HEAD и main сохранены; commit/push не выполнялись.
