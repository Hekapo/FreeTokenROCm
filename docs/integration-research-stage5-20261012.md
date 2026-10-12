# Этап 5: подготовка AMD 10.1 / Torch 2.14 и проверка компилятора

Дата: 12.10.2026. Ветка: `integration/research-refresh-20261002`; HEAD: `e5e2899eef36e55eb84a58a4cd59de8766ac8c46`.

**Статус: окружение подготовлено, статические/CPU/native gates пройдены; новый комплект не принят.** Triton Windows3.8.0.post29 воспроизводимо ошибается в histogram и loop range analysis. GPU operators, model smoke и long/Stop нового стека не выполнялись.

## 1. Отдельный candidate и точные зависимости

Candidate: `C:\Users\Iskandar\FreeTokenROCm\.venvs\freetoken-rocm1010-triton38post29-py312`, CPython3.12.0. За основу взяты Python dependencies, принятые в [этапе4](integration-research-stage4-20261012.md).

| Пакеты | Control / этап4 | Candidate |
|---|---|---|
| Torch, amd-torch-device-gfx12-0, amd-torch-device-gfx1201 | 2.13.0+rocm10.0.0 | 2.14.0+rocm10.1.0 |
| rocm, SDKcore/devel/libraries/device-gfx1201 | 10.0.0 | 10.1.0 |
| triton-windows | 3.8.0.post28 | 3.8.0.post29 |
| Остальные65 packages | Принятые версии этапа4 | Те же версии |

Все74 версии зафиксированы в [candidate constraints](../constraints/windows-rocm1010-candidate-20261012.txt). [Artifact manifest](../constraints/windows-rocm1010-artifacts-20261012.json) содержит filenames, exact URLs, sizes, tags и SHA256.

Свежая venv создана `uv` от того же Python. Из этапа4 независимо скопированы12158 файлов65 других пакетов, 345977780bytes; file hashes совпали, hardlinks к исходной среде не создавались. Старые vendor distributions не копировались. Девять новых packages установлены из local wheels с explicit `--python`, `--no-config`, `--offline`, `--no-build`, `--link-mode copy` и полными constraints. Console scripts старой среды не переносились; FreeToken импортируется из нужного checkout через PYTHONPATH.

Скачаны девять exact artifacts, суммарно1847288771bytes (1,72GiB). Windows wheels подходят CP312/win_amd64 либо py3/Windows. AMD index не публикует независимый SHA256: записанные AMD hashes идентифицируют скачанные HTTPS bytes. SHA256 Triton wheel дополнительно совпал с publisher value в PyPI. [AMD index](https://stable.repo.amd.com/rocm/whl-next/), [официальная установка](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).

`rocm==10.1.0` поставляется как sdist. До исполнения просмотрены setup.py и helper, выбирающий target family. Local wheel собран отдельно, offline, без build isolation, с setuptools84.0.0 и explicit `ROCM_SDK_TARGET_FAMILY=gfx1201`; GPU autodetection не требовался. Generated metadata проверены отдельно от PKG-INFO исходного архива: generic `device` extra теперь указывает gfx1201.

| Artifact | SHA256 |
|---|---|
| rocm-10.1.0.tar.gz | `e6616e62ebd1324031681e6e59349d46ee4f143a8d8f4f901a91c35c993041fc` |
| Local rocm-10.1.0-py3-none-any.whl | `d714cdf643962d1006c028a7694a98e67963959318e5150a526f1703d95afcb2` |
| triton_windows-3.8.0.post29-cp312-cp312-win_amd64.whl | `937ba2ee53b6e664709898537de3eb9eab427a498c04da58f2bd42e1af93e298` |

SDK developer tar расширен штатным `rocm_sdk._devel.get_devel_root()`. Удаление внутреннего `_devel.tar` этим helper учтено в проверке RECORD;9088 развернутых файлов/links отдельно сверены с publisher archive и matching SDK components.

## 2. Изменения source и проверка совместимости

В `pyproject.toml` обе границы Torch изменены с `>=2.11,<2.14` на `>=2.11,<2.15`: runtime и build isolation получают одинаковый диапазон. Перед изменением candidate2.14 отвергался; после него2.11/2.13/2.14 разрешены,2.15 запрещён. CUDA source selection из tool.uv.sources не использовался для AMD candidate.

Проверено:

- `uv pip check`: все74 installed packages совместимы.
- 103 active Requires-Dist, включая explicit Torch device и ROCm extras, удовлетворены; project mandatory dependencies также удовлетворены.
- Torch/SDK/Triton/TVM imports принадлежат candidate; FreeToken принадлежит текущему checkout.
- Torch сообщает2.14.0+rocm10.1.0, фактический HIP version string —7.16.26385. Triton module version3.8.0 отличается от distribution post29; обе версии сохранены.
- Сигнатуры используемых cpp_extension/hipify helpers сохранены. `IS_HIP_EXTENSION=True`; Windows link/hipify adapters совместимы с этим wheel.
- **298 CPU tests passed,2 skipped,1 existing Starlette-httpx warning**,32,13s. Включены parser/tokenizer/API contracts этапа4, sampling penalties, cache rebuild ordering, KV accounting, backend probes и FLA interpreter padding/reference. Два skips относятся к sampler/GDN GPU gates; их нельзя считать пройденными.

## 3. Native build/load и TVM-FFI

Все native/JIT outputs созданы в отдельных каталогах этапа5. Checkout binaries и kernels прежних окружений не заменялись. `MAX_JOBS=1`; visibility variables=-1; `torch.cuda.is_initialized()` оставался false.

| Проверка | Результат |
|---|---|
| Native HIP GGUF module | Свежий build и import прошли;88,968s, шесть ggml exports |
| `_pinned_tensor` | Свежая сборка setup.py в отдельный build-lib и import прошли |
| `_cpu_moe` | Свежая сборка/import прошли; CPU capability marker max_generic_act_id=4 |
| Оба host extensions | Build/load23,157s, source checkout headers и matching Torch libs |
| TVM-FFI0.1.13.post3 | Собран CPU C++ module с текущим freetoken/tensor.h; три real Torch tensor cases совпали с FP64 reference, включая storage offsets8/17; invalid dtype/strides отвергнуты |

Для TVM-FFI сохранён baseline `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`; результат относится к обычному CPU DLPack пути. GPU interop и отключённый Torch C API этим тестом не квалифицированы. Синтаксис сверён через Context7 и [официальный TVM-FFI guide](https://github.com/apache/tvm-ffi/blob/main/docs/guides/python_lang_guide.md).

SDK compiler: AMD clang24.0.0git, ROCm/llvm-project revision `d6f6cb691863e9258034ab48c1dfce21689d5606`. Версия, build commands, generated ninja, output hashes и106 фактически загруженных DLLs сохранены. HIP/rocBLAS/Torch DLLs взяты из candidate, смешения со старой venv не обнаружено.

| Binary | SHA256 |
|---|---|
| freetoken_gguf_kernels.pyd | `91d15fac5c4fc9b974fa30303f714bad95a2015dbe4282c25f0bab16fa54d624` |
| `_pinned_tensor.cp312-win_amd64.pyd` | `76b92c6bded53eaf835a6462dfb53735f4f9c8c69ae91cf04e07a48c666f29c4` |
| `_cpu_moe.cp312-win_amd64.pyd` | `4d589c63c17453eaac9b8ead02cd7628fc632eb74d949acb7218fae23cbe9f2d` |
| stage5_tensor_abi.dll | `4436a2d7e120a9f989df05d7a3f41fdff4995bde1d14a78cc0e594089c62d5e7` |

GGUF build выдавал LLVM warnings о невозможности unroll для части loops; compile/link/import завершились успешно. Численная GPU-корректность native operators ещё требует отдельного gate.

## 4. Воспроизведённая ошибка Triton post29

Сверены текущие primary PR metadata и diffs:

- [11577](https://github.com/triton-lang/triton/pull/11577), head `55a1abfc43d8947068c3af1ae14c02856ddf99c7`: histogram backport11246 в release/3.8.x остаётся open/unmerged.
- [11429](https://github.com/triton-lang/triton/pull/11429), head `ee3956429ace3e7f38ba6133e0df795315c722be`: loop backport11313 закрыт без merge.

В установленном libtriton выполнен `tritonamdgpu-fold-true-cmpi` над тремя маленькими MLIR examples из этих regression tests:

| Example | Требуемый результат | Полученный результат |
|---|---|---|
| Histogram positive-count comparison | Сохранить `hist > 0` | Постоянный dense false |
| Dynamic loop0/1 iteration | Сохранить зависимость от program_id | Постоянный false |
| Nested loop with widening inner bound | Последний count=4, `count !=4` false | Постоянный true |

Затем полный pipeline `triton.compile(ASTSource(...), target=GPUTarget('hip','gfx1201',32))` скомпилировал histogram examples1024/64 и4096/128. В обоих TTGIR predicate у `amdg.buffer_store` стал постоянным false: output write подавлен для любого входа. LDS соответственно256 и512bytes, в пределах64KiB; это не устраняет ошибку результата. Сохранены TTIR/TTGIR/LLIR/AMDGCN/HSACO. GPU driver.active не использовался, handles не инициализировались, kernels не запускались.

**Причина ограничена конкретным compiler pass.** В свежем отдельном процессе временное отключение добавления `tritonamdgpu-fold-true-cmpi` восстановило динамический histogram predicate в обоих full-pipeline examples. Другие параметры и kernels сохранены. Это диагностическая проверка причинности. Installed Triton files и production initialization не менялись; numeric GPU/reference, loop full-pipeline и performance acceptance этого bypass отсутствуют.

Вывод относится к данным kernels и range-analysis pass. Связь с прежним NaN или зависанием ПК не установлена. Kernel interpreter/CPU checks не подтверждают корректность машинного кода этого compiler.

Начальный harness вызвал pass_manager.run без обязательного второго аргумента текущего wheel; этот тестовый вызов исправлен по installed compiler API. Для buffer-ops TTGIR detector учитывает `amdg.buffer_store` и его false predicate. Промежуточные логи сохранены; итоговый исходный compiler gate — `compiler-reproducers-r2`, диагностический bypass — `compiler-pass-bypass`.

## 5. Host compatibility и решение о runtime

На ПК: Windows11 Pro24H2, build26100; RX9070XT/gfx1201; DriverStore32.0.31041.1004. [ROCm10.1 matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) указывает25H2 и Adrenalin26.10.41.05; [driver release](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html) сообщает DriverStore32.0.31041.5005. Поддержка ОС самого driver package шире (Windows11 21H2+), поэтому это различие с квалифицированной ROCm row, а не доказательство невозможности загрузки драйвера на24H2.

Shared driver и ОС требуют отдельного обслуживания; venv их не изолирует. Нагрузочные проверки нового candidate остановлены на failed compiler gate. GPU operators, обычный long64K/Stop и новые TPS measurements отсутствуют. Приёмка64K на прежнем квалифицированном стеке остаётся описана в этапах2–4.

Для принятия нового vendor stack остаются:

1. Compiler build с проверенными histogram11246 и loop11313 fixes либо отдельно спроектированный ограниченный обход. Повторить эти CPU compiler examples и зафиксировать новые artifacts/compiler cache namespace.
2. Зафиксировать фактическую согласованность ОС/driver с выбранной ROCm row. Системное обновление оформляется отдельно; затем актуализировать control на новом shared driver. Для сравнения на прежней комбинации требуется явная маркировка её отклонения от vendor row.
3. Selected GPU sampler/GDN/Q4_1 operators, model-geometry LDS и CPU/GPU reference. Затем один managed ordinary long64K/Stop с45GiB launch free commit и неизменными12/4/3GiB runtime resource floors.
4. Принимать candidate по корректности и полезному результату. Выбирать окружение и его отдельные binary/cache roots; обычный launcher не переключён на10.1.

Portable source initializer этапа6.1 можно разрабатывать на уже квалифицированном ROCm10.0 runtime независимо от принятия10.1. Оно не закрывает compiler correctness gate этапа5.

## 6. Сохранность и raw evidence

Hash verification:33718 файлов original control без изменений;12158 copied base files без изменений и совпадают в candidate;13890 installed vendor RECORD files совпали;9088 expanded SDK files/links сверены;56 прежних repository files не изменены. HEAD/branch прежние; diff check прошёл. Source diff этого этапа — две Torch bounds; snapshots/report учитываются отдельно.

Raw audit: [stage5 directory](C:/Users/Iskandar/FreeTokenROCm/rocm10-20260926/research-refresh-20261011-audit/stage5-20261012).

Ключевые записи: baseline/before, artifacts-selected/verified, generated metapackage metadata/build log, vendor resolve/install, candidate-packages, import-gate, CPU XML, native-gate/loaded-dlls, TVM gate, compiler before/after IR и HSACO, verification, managed status. Reproducers и compile-only launcher лежат рядом с audit и в triton38-lab соответственно. Подготовленная среда, downloaded artifacts и временный uv-install-cache сохранены: очистка последнего отклонена автоматической проверкой разрешений с причиной `blocked by policy`.
