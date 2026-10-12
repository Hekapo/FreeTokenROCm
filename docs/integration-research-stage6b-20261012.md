# Этап 6.2: загрузка выбранных экспертов для короткого prefill

Дата: 12.10.2026. Ветка: `integration/research-refresh-20261002`, HEAD
`e5e2899eef36e55eb84a58a4cd59de8766ac8c46`; изменения этапа не закоммичены.

## Выбор расхода

Active path: `Qwen3_5MoE.forward` → `OffloadMoELayer.prefill_forward` →
`_prefill_routed` → `OffloadMoeCache.materialize_layer/copy_missing` → native
`fused_experts_gguf`. При отключённом overlap прежний путь переносит все 256
экспертов каждого слоя. Startup logs подтверждают 80 pinned host banks / 18,75GiB
для Qwen3.6-35B-A3B-Q4_1. Полная загрузка слоя повторяется даже при малом числе
token rows; kernel вычисляет только маршруты из top-k.

Для actual prompt24/top-k8 число выбранных экспертов слоя не больше192, поэтому
полная загрузка переносит как минимум64 неиспользуемых эксперта. Для 40 слоёв
это минимум25% объёма экспертных banks; точный объём H2D model run не измерялся.
Повторяющиеся ID и уже resident experts дополнительно уменьшают misses.
Это доказательство лишней работы из active source path и geometry; скорость
проверяется отдельно ниже.

Идея и публичное имя CLI взяты из [donor ff4966](https://github.com/tomasuz/FreeToken/commit/ff4966bd3b57de03f1163967d6c07de2a41d7ee9),
Apache2.0. Порт адаптирован к нашему native Q4_1 dispatch и ограничен проверенными
условиями. Donor NVIDIA/NVFP4 результаты не используются как наши.
Context7 не нашёл соответствующий flashlib после двух resolve queries; contract
проверен по установленному flashlib0.3.0 и [первичному source](https://github.com/FlashML-org/flashlib/blob/main/flashlib/kernels/slot_cache/triton/lru_ensure.py).
Admission разрешает дубликаты и in-place slot rewrite; caller гарантирует достаточную capacity.

Attention632, bounded MoE workspace, dense/router GEMV и GDN fusion не переносились:
для текущего выбора нет отдельного доказательства, что они дают более полезное
улучшение этого workload. Они остаются условными кандидатами, без обещанных gains.

## Реализация

Добавлен `--moe-prefill-demand-tokens`, default0, допустимые значения0–32.
Настройка проходит через ServerArgs/EngineConfig в OffloadMoeCache; dense-model
normalization сбрасывает её в0. EngineConfig и direct cache constructor отклоняют
нецелое/отрицательное значение и значение>32 до cache allocation.

Путь активен только при native Q4_1 `(3,3)`, ROCm GPU, GPU decode, pinned GPU layer,
отключённом prefill overlap, положительном числе token rows внутри порога и
`0 < raw_routes < num_experts`, `raw_routes <= min(cache_size,256)`.
Последний предел ограничивает quadratic dedup block flashlib. Admission не
вызывает `unique().item()` или host synchronization для определения пригодности.
Остальные случаи используют прежний full-layer/overlap path.

Выбранные experts проходят через существующие `ensure_experts`/`copy_missing`;
GGUF kernel получает те же packed banks по новым slot IDs, `is_prefill=True`.
Нативные GEMV, routing scores, weights, activation и reduction не изменены.
Prefill admission не увеличивает decode statistics и decode-frequency histogram.
Нового workspace или host banks нет.

Qualified opt-in:

```powershell
ft serve --model C:/models/Qwen_Qwen3.6-35B-A3B-Q4_1.gguf `
  --moe-prefill-demand-tokens 32 --disable-moe-prefill-overlap
```

Это пример настройки экспертного пути; остальные параметры 64K берутся из
сохранённого проверенного профиля. Default0 сохраняется. Откат: убрать новую
опцию либо задать0 и перезапустить процесс; overlap настраивается отдельно.

## Проверки

- До реализации новый тест падал на отсутствующем `prefill_demand_tokens`.
- **42 CPU tests passed / 5 skipped, 9,90s**: default/threshold validation,
  dtype/backend/profile gates, remapped slot views, дубликаты, сохранность decode
  counters, CLI parsing, прежние offload contracts и daemon import safety.
- Из этих5skips два новых operator cases запущены отдельно на GPU: **2 passed,
  92,85s**, включая fresh native build. Один warning от setuptools private API.
- Geometry `(experts,slots,tokens,top-k)`: `(16,24,3,4)` и `(256,2048,31,8)`.
  Три successive demand calls в каждом случае совпали с full native Q4_1
  **побитно**. Проверены hits без новых copies, duplicates, other-layer eviction,
  возврат после full-layer materialize, copy bounds и unchanged decode counters.
- Малый случай также сверён с независимой FP32 dense/activation reference:
  rtol0,02 / atol0,1. Admission LDS:64 и1024bytes, ниже64KiB gfx1201 limit.
- Workload tokenizer на main/current совпал: actual24tokens, renderer SHA256
  `a06fcbb79801c1f9c080b46b29018a2e91ab34c53e46c4482779de48649f2281`.
  Spec keys и все три client phases проверены офлайн; simulated timings не используются.

Промежуточный CPU test ошибочно импортировал несуществующий `add_server_args`;
исправлен на реальный `parse_args`. Это ошибка fixture, production исправления
для неё не потребовались. После GPU gate добавлена только более строгая shape
проверка `raw_routes<=256`; проверенные GPU branches остались теми же.

## Сравнение модели

Все три phases: control73 packages, AMD Torch2.13.0+rocm10.0.0 / SDK10.0.0 /
Triton Windows3.8.0.post28, прежние OS/driver. Candidate10.1 этапа5 не использовался.
Configured context/KV65536, BF16 KV / FP32 GDN, prefill512, MoE2048/LRU,
serial expert load, concurrency1, graphs0, naive cache, overlap **off** во всех
трёх phases. Buffer ops=true; source knob initializer и lab knob pinning off.
Temperature0, thinking=false, output budget512, request/no-progress45s,
periodic logits safety every64. Каждый запуск имеет отдельные native/Triton caches,
замороженные source/package/protocol inputs и owned observer/Stop.

Один warmup исключён, затем два measured requests на каждый вариант:

| Source / demand threshold | TTFT, медиана, s | Два measured TTFT, s | Decode TPS, медиана | Два measured TPS | Total, медиана, s |
|---|---:|---|---:|---|---:|
| main `c4832f35055c`,0 | 0,8205 | 0,812 / 0,829 | 40,52 | 40,45 / 40,59 | 10,4455 |
| current,0 | 0,8280 | 0,828 / 0,828 | 40,33 | 41,60 / 39,06 | 10,5075 |
| current,32 | 0,2500 | 0,250 / 0,250 | 39,53 | 39,81 / 39,25 | 10,1170 |

ON/OFF: **TTFT−69,81%** (примерно3,31×), decode−1,99%, total−3,72%.
ON/main: TTFT−69,53%, decode−2,45%. Выигрыш TTFT существенно больше observed
spread и client clock resolution. Ускорение decode не подтверждено; его отличие
меньше6,29% spread контрольных measured requests. Total уменьшился в этой
небольшой выборке; устойчивый небольшой gain не установлен.
OFF/main включает предыдущие интеграции и не изолирует только этот этап.

Все9successful requests: prompt24, completion391, пустой reasoning,
`finish_reason=stop`, одинаковые response bytes. Content SHA256:
`34a4f1e5bb080915a30b7f67a8546b8e72da130622436caa0fcb81a2eb62c0ee`.
Decode TPS считается как `(completion_tokens−1)/(finish_event−first_output)`;
первый фрагмент `'1'` является одним token. Завершающий tail включён; первый token
и его время исключены. Python3.12 Windows monotonic resolution около15,6ms.

Это короткий запрос при capacity64K, не decode после заполнения64K.
Последовательность main→off→on, n=2 на вариант; она не оценивает длительную
устойчивость. Ввод/вывод отличаются от6.1 (24/391 вместо34/292), поэтому их
скорости не сравниваются между собой. Matched4K/8K series и long64K не повторялись.

## Ресурсы и Stop

Минимумы за весь запуск:

| Phase | Free RAM, GiB | Free commit, GiB | Free local VRAM, GiB |
|---|---:|---:|---:|
| main | 25,232 | 13,120 | 6,845 |
| off | 25,903 | 13,087 | 6,860 |
| on | 25,861 | 13,048 | 6,854 |

Launch commit floor45GiB и runtime floors12/4/3GiB сохранены. Все три Stop:
root/observer отсутствуют, tree0, listeners0, recovery_required=false.
После последнего: RAM51,054 / commit47,300 / local VRAM15,046GiB.
Historical quarantine64K сохранён. Новых failed model requests или startup
recovery не было. Supervisor exit сообщения после Stop отнесены к shutdown
по сохранённым `stop-intent.json` log offsets; до этих offsets таких ошибок нет.

Raw runs: [main](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/ctx64k-20261012-181602-0c856396),
[off](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/ctx64k-20261012-182013-19e1a3fb),
[on](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/results/ctx64k-20261012-182539-fcfd6c17).

## Решение

Этап6.2 завершён для opt-in короткого ROCm Q4_1 prefill: functional gates пройдены,
TTFT benefit наблюдается в проверенном profile. **Default остаётся0**; включение
в ordinary overlap profile не принято. Полная длинная серия не нужна для
этого ограниченного acceptance, но обычная длительная эксплуатация нового
профиля здесь не квалифицирована. Stage5 compiler acceptance остаётся failed.

После runtime изменены документация и описание fallback в comments/docstring
`_prefill_routed`; executable AST последнего сравнивается с сохранённым runtime
source без docstrings. Остальные accepted source и installed control files
проверяются по bytes. Следующий шаг — этап8: итоговый
срез принятых changes и оставшихся gates, затем минимальный matched capacity
comparison при необходимости. Optional prefix reuse этапа7 не блокирует этот срез.

Evidence: [audit](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/stage6b-20261012),
[runtime comparison](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/stage6b-20261012/runtime-comparison.json).
