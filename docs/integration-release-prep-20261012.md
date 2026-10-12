# Обычный запуск 64K: итоговое ревью и подготовка интеграции

**12.10.2026: обычный managed 64K подключён и проверен.** Лаунчер выбирает `64k` по умолчанию. Принятые Python-библиотеки этапа4 используются с прежним native ROCm10.0 / AMD Torch2.13 / Triton post28. Новые source-опции этапа6 остаются выключенными. Слияние и сохранение Git-коммитов в этом этапе подготовлены; команды commit/push/merge не выполнялись.

## Что изменилось

- `Start` и `Preflight` выбирают reviewed profile: `.venvs/freetoken-pydeps-20261012-py312`, context/KV65536, backend/frontend concurrency1, prefill512, MoE2048/LRU, serial expert load, BF16KV/FP32GDN, graphs0, ordinary prefill overlap, stable AMD knob. Другие явно выбранные профили и диагностические планы сохраняют свои настройки.
- Новый source verifier проверяет **476 production files, 2 build files, 74 package versions и 5 ключевых native/compiler binaries** без загрузки GPU. Документация и переход файлов из untracked в staged/committed не меняют идентичность проверенного кода. Изменение кода, build inputs, пакетов, бинарников или набора Python-файлов блокирует запуск.
- Working-release schema2 связывает 24 runtime files, stage8 matched9requests, source/native verification и прежний ordinary63488-token long/Stop. После пилота добавлена запись обычного запуска. Legacy schema1 с602/62/3 сохраняет прежние правила; эти числа не приписаны новой серии.
- Source flag0 / demand0, AMD buffer ops1, logit safety every64. При обычном запуске отключены per-token trace, finite snapshots и scheduler/host profiling. Startup free commit45GiB, runtime RAM/commit/localVRAM floors12/4/3GiB, восстановление commit28GiB сохранены.
- Исторический карантин остаётся на диске. Активный профиль разрешён на основании проверяемой schema2 записи. Новые recovery faults и изменения квалифицированного кода по-прежнему блокируют Start.

## Итоговое ревью

Проверены source changes этапов1–6: FTW guard, penalties/stream ordering, cache sizing/rebuild, FLA/int64/padding/native Q4, diagnostics default-off, tokenizer/API adapters и bounded short-prefill opt-in. Проверены relevant tests и сохранённые CPU/GPU/model gates. Подтверждённых новых дефектов, требующих изменения production source, не найдено. Все476 файлов кода byte-identical к stage8 acceptance; повтор всей CPU/GPU серии не выполнялся.

Новый release gate: **18 positive/negative cases**, включая отсутствующие/изменённые runtime inputs, неверные capacity/venv, плохие output/Stop и отсутствующую long/source proof. Новый source verifier: **9 CPU contracts**, включая изменение code/build/packages/native, добавление Python-файла и устойчивость к docs/staging. Архивная версия reader отвергала новый профиль до изменения. Default64K и сохранение диагностических args проверены отдельно. Ordinary preflight прошёл.

## Одна прикладная проверка обычного запуска

`Start -Profile 64k` → один контрольный запрос → idle → `Stop`. Как и прежде, Start дополнительно выполняет штатный readiness request: перед нашим запросом completed=1, после=2. Это **один прикладной запрос и одна штатная startup-проверка**, без новой benchmark/soak серии.

| Проверка | Результат |
|---|---|
| Обычный прикладной запрос | Pass; prompt24, completion391, правильные числа1…100, finish=stop, reasoning пуст |
| TTFT / total | 1.063s / 9.063s |
| Выход | SHA34a4f1e5… совпал со stage8; same instance после запроса |
| Idle | KV used_pages0, active requests0, capacity65536 |
| Min free RAM / commit / local VRAM | 26.032 / 14.008 / 6.918GiB |
| Наблюдатель / stable knob | 54 samples; 3 processes, Torch imported first, buffer ops true, finite-dir отсутствует |
| Stop | STOPPED; root/observer/tree/listener0; recovery=false |
| Release status | valid=true; active quarantine=false; historical quarantine retained=true |

Этот один cold application request не используется для нового speed gain. Для сравнения capacity сохраняется [серия этапа8](integration-research-stage8-20261012.md): 4K37.20 / 8K37.42 / 64K37.56TPS. Полное итоговое дерево на длинном prefix не прогонялось повторно; inherited stage2 long имеет указанную в отчётах область, numerical14files и native identity сохранены. ROCm10.1/post29 остаётся неквалифицированным, prefix reuse7 отложен.

## Как запускать

Из PowerShell7:

```powershell
& C:\Users\Iskandar\FreeTokenROCm\rocm10-20260926\triton38-lab-20260928\service38_control.ps1 -Action Preflight
& C:\Users\Iskandar\FreeTokenROCm\rocm10-20260926\triton38-lab-20260928\service38_control.ps1 -Action Start
& C:\Users\Iskandar\FreeTokenROCm\rocm10-20260926\triton38-lab-20260928\service38_control.ps1 -Action Status
& C:\Users\Iskandar\FreeTokenROCm\rocm10-20260926\triton38-lab-20260928\service38_control.ps1 -Action Stop
```

Профиль по умолчанию —64K; для явного выбора используйте `-Profile 64k`. Адрес `http://127.0.0.1:1919`. Перед запуском нужен45GiB free commit. Новая source или package комбинация требует нового reviewed record; verifier не отключать.

## Артефакты и подготовленные коммиты

Audit: [release-prep-20261012](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/release-prep-20261012); `baseline.json`, `release-gates.json`, `source-gates.json`, `ordinary-summary.json`, `ordinary-analysis.json`, `working-release.json`, `reviewed-source.json`, `final-status.json`, `lab.patch`, `docs.patch`, `verification.json`, `commit-plan.json`.

Рабочие lab files находятся вне Git checkout. Их полный [snapshot](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/release-prep-20261012/reviewed-service38-runtime.zip) и `runtime-current/` сохранены рядом с checksum record; Git-коммиты engine/documents сами по себе эти локальные файлы не содержат. В `commit-plan.json` перечислены отдельные paths и messages для code/tests и docs/constraints; `.serena/` исключена. Prospective Git tree собран через temporary index, реальный index сохранён. Main остаётся чистым наc4832f35; текущая ветка `integration/research-refresh-20261002` остаётся наe5e2899e с незакоммиченными изменениями.

Откат обычного launcher: остановить owned service и восстановить четыре изменённых lab files из `lab-before`/`lab.patch`; новые record/source helper сохранить в audit. Возврат dependency среды выбирает original73 control и его caches. Старый quarantined64K не станет разрешённым от одного восстановления старых scripts: его legacy release gate сохраняется.
