# Подготовка входов для валидации

`libs.validation.datasets.input_adapters` содержит `ArrayInputAdapter`, `BuoyInputAdapter`, `PreparedInputs`, `ComparisonContext` и `make_metric_field`. Адаптер согласует уже прочитанные входы. Чтение файлов, пространственная интерполяция и выбор ближайшего наблюдения остаются задачами датасетов.

## Существующая сеточная валидация

Обычный вызов `Validator` сохраняется:

```python
from libs.validation.validator import Validator
from libs.validation.metrics import MSE
from libs.validation.aggregators import AverageAggregator

validator = Validator(
    datasets=[grid_model, grid_reference],
    metrics=[MSE()],
    aggregators=[AverageAggregator()],
)
validator.run(dates=dates)
summary = validator.summarize()
```

По умолчанию используется `ArrayInputAdapter`. Он передаёт NumPy-массивы метрике без изменения осей, масок и правил broadcasting; подклассы `ndarray` сохраняются. Метаданные `.meta` объединяются в порядке входов, как раньше: последний вход имеет приоритет при совпадении ключей.

Непустой массив, целиком заполненный `NaN`, по-прежнему передаётся метрике: некоторые метрики определяют собственное поведение для пропусков. Пустые массивы пропускаются. `BuoyBatch` требует явного `BuoyInputAdapter`.

## Валидация на буях

Ниже `model_on_buoys` — датасет с модельными значениями в точках наблюдений и описанием осей, приведённым в следующем разделе. `observations` — `BuoyObservationDataset` с заданным `name`; имена датасетов должны различаться.

```python
from libs.validation.datasets.input_adapters import BuoyInputAdapter
from libs.validation.aggregators import AverageAggregator, RawFieldAggregator

validator = Validator(
    datasets=[model_on_buoys, observations],
    metrics=[MSE()],
    aggregators=[AverageAggregator(), RawFieldAggregator()],
    input_adapter=BuoyInputAdapter(
        reference_index=1,
        variables=("ice_thickness",),
    ),
)
validator.run(dates=dates)
summary = validator.summarize()
```

`reference_index` относится к порядку аргументов конкретного вызова метрики. Фиксированный `reference_index=1` требует, чтобы наблюдения были вторым аргументом каждой сравниваемой пары. Для совместного запуска унарных и парных метрик со списком `[model_on_buoys, observations]` используйте `reference_index=-1`: в паре опорным будет наблюдение, в унарном вызове — единственный вход с собственным описанием точек. Если датасетов больше двух, задайте `combinator`, который включает наблюдения в каждую сравниваемую пару на выбранной позиции: обычные сочетания могут создать пару двух моделей или поставить наблюдения на другую позицию.

Опорный вход задаёт идентификаторы буёв и времена проверки. Адаптер приводит остальные входы к `(N,T,V)`, переставляет буи, времена и переменные по их меткам, заполняет отсутствующие буи и сроки пропусками. Совпадение времени проверяется точно; дополнительного `nearest` здесь нет. Неоднозначные дубли меток и отсутствие запрошенной переменной вызывают ошибку.

Во всех массивах метрики применяется одна маска: нужны конечные значения всех сравниваемых входов, разрешающие маски измерений и допустимая позиция наблюдения. Исходные массивы не изменяются. Если общих пригодных отсчётов нет, метрика и агрегаторы для этой выборки не вызываются; итог для неинициализированного агрегатора — `None`.

## Как передать модельные значения

`BuoyBatch` уже содержит необходимое описание. Для массива модели используйте существующий `MetricField`:

```python
import numpy as np
from libs.validation.validator import MetricField

# obs — тот BuoyBatch, в точках которого рассчитаны model_values.
# model_values имеет форму (T, V, N); его каналы здесь hi и hs.
prediction = MetricField(
    model_values,
    dims=("time", "variable", "buoy"),
    bids=obs.bids,
    datetimes=obs.datetimes,
    var_names=("hi", "hs"),
    units=("cm", "cm"),
    coords=obs.coords,
    valid=np.isfinite(model_values),
    coord_valid=obs.coord_valid,
)

adapter = BuoyInputAdapter(
    reference_index=1,
    variables=("ice_thickness",),
    variable_maps={0: {"ice_thickness": "hi"}},
    target_units={"ice_thickness": "m"},
)
prepared = adapter.prepare([prediction, obs], metric=MSE())
```

Обязательные метаданные массива: `dims`, `bids`, `datetimes`, `var_names`, `units`. `dims` — перестановка `("buoy", "time", "variable")`; значения и `valid` имеют одинаковую форму и порядок осей. `coords` всегда имеет форму `(N,T,2)` с порядком `[latitude, longitude]`, а `coord_valid` — `(N,T)`, независимо от `dims`.

Координаты и маски модели необязательны. Если координаты переданы, адаптер проверяет совпадение с позициями опорного входа с допуском `1e-5` градуса и учётом эквивалентных долгот. Отсутствие координат модели означает, что вызывающая сторона гарантирует пространственное сопоставление. Опорный вход обязан содержать координаты.

В `variable_maps` внешний ключ — индекс входа, внутреннее соответствие — **каноническое имя → имя в этом входе**. По умолчанию имена совпадают. `target_units` задаёт конечные единицы; без него используются единицы опорной переменной. Поддерживаются известные преобразования масштаба, например `cm → m`. Несовместимые единицы вызывают ошибку.

### Интерполятор и унарные метрики

При работе с новым `BuoyBatch` обёртка `InterpolatedOverBuoysDataset` возвращает `MetricField` с описанием точек из того batch, по которому выполнена интерполяция. Форма модельного результата сохраняется, например `(T,V,N)`, а `dims` сообщает адаптеру порядок осей. Параметр `array_dims` для такого результата не нужен.

```python
from libs.validation.datasets.utils import InterpolatedOverBuoysDataset
from libs.validation.metrics import IdentityStat, MAE, Difference

model_on_buoys = InterpolatedOverBuoysDataset(
    model_raw,
    observations,
    base_time_axis=0,
    buoy_time_axis=1,  # BuoyBatch.coords имеет форму (N,T,2).
)
adapter = BuoyInputAdapter(
    reference_index=-1,
    variables=("ice_thickness",),
    assume_aligned=True,
)

validator = Validator(
    datasets=[model_on_buoys, observations],
    metrics=[IdentityStat(), MAE(), Difference()],
    aggregators=[AverageAggregator(), RawFieldAggregator()],
    input_adapter=adapter,
)
validator.run(dates=dates)
```

`assume_aligned=True` здесь разрешает взять отсутствующие имена и единицы модельных каналов из описания наблюдений, сохранённого в `meta["buoy_sampling"]`. Это явное утверждение вызывающей стороны о совпадении **всех исходных каналов**, их порядка и единиц. Если модель выдаёт один канал, а batch содержит толщину льда и снега, такое предположение неверно. Имена и единицы, явно заданные моделью, всегда имеют приоритет.

Чтобы не делать это предположение, задайте модельную схему в обёртке: `InterpolatedOverBuoysDataset(..., var_names=("ice_thickness",), units=("m",))`. Тогда `assume_aligned=True` не нужен. Обёртка также использует явно заданные `var_names` и `units` из метаданных модельного поля.

Обёртка сохраняет ID, времена, координаты, их валидность и исходные времена позиций. Некорректные позиции возвращаются как `NaN`, пустой batch — как пустое поле. Маска измерений наблюдения не переносится в модель: `IdentityStat()` модели считает её собственные доступные значения в точках, а парные метрики используют общую маску модели и наблюдений. Неопределённость и исходные времена измерений наблюдения не выдаются за характеристики модели. Обёртка выполняет пространственное сопоставление по соответствующим временным индексам; её настройка не добавляет временной интерполяции.

Для старых обёрток, возвращающих полностью неразмеченные массивы, остаётся режим `assume_aligned=True, array_dims=("time", "variable", "buoy")`. Он требует отдельного размеченного опорного входа в том же вызове метрики, поэтому не решает унарный вызов на голом модельном массиве. Произвольное частичное описание точек по-прежнему отклоняется; сведения о запуске и сроке прогноза сохраняются.

## Метрики, результат и агрегирование

Для скалярных метрик компоненты маскируются независимо. `component_policy="joint"` требует пригодности всех выбранных каналов в одной паре «буй–время». По умолчанию `component_policy="auto"` выбирает совместную маску для известных векторных метрик.

Для векторных данных используйте `AngleError(var_axis=-1)` или `VectorNorm(var_axis=-1)`; компоненты должны иметь одинаковые единицы. `VectorNorm` по умолчанию имеет один вход, поэтому для самостоятельного вызова ему нужен адаптер с `reference_index=0` или `-1`. Результат векторной редукции имеет `(N,T)`. Пространственные метрики с сеточными окрестностями автоматически не становятся применимыми к буям.

`make_metric_field` сохраняет ошибки как `MetricField` и добавляет корректное описание результата. Для MSE толщины:

```python
from libs.validation.datasets.input_adapters import make_metric_field

metric = MSE()
errors = metric.compute(*prepared.arrays)
field = make_metric_field(
    errors,
    metric=metric,
    context=prepared.context,
    input_dims=prepared.input_dims,
)
assert field.meta["dims"] == ("buoy", "time", "variable")
assert field.meta["units"] == ("m^2",)
```

Контекст включает `bids`, `coords`, `datetimes`, итоговую `valid`, входные `input_var_names` и `input_units`. Метаданные отдельных входов сохраняются раздельно в `field.meta["inputs"][i]["metadata"]`: например, `init_time`, `valid_time` и `lead_time` модели не перезаписывают сведения наблюдений. Выровненные `value_times`, `coord_times` и `uncertainty`, если они были доступны, находятся непосредственно в `field.meta["inputs"][i]`.

Для пользовательской редукции задайте `result_dims` и `result_units` в адаптере, например `result_dims=("buoy", "time")`. Неизвестным метрикам единицы автоматически не приписываются. Форма результата проверяется по объявленным осям; агрегирование выполняется отдельно.

`AverageAggregator` пригоден для общего среднего одной выбранной скалярной переменной: он суммирует все конечные ошибки и считает число элементов, поэтому каждый отсчёт имеет одинаковый вес. При MSE его результат — средняя квадратичная ошибка, не RMSE. `RawFieldAggregator` сохраняет полные поля с метаданными по датам и допускает меняющееся число буёв.

Сеточные `SpatialAggregator` и региональные агрегаторы нельзя автоматически применять к перемещающимся буям: постоянный индекс строки не означает постоянный прибор или положение. Новые агрегаторы по `bid`, региону и сроку прогноза в это изменение не входят. Перекрывающиеся окна сами по себе не устраняют повторный учёт наблюдений.

## Окружение

Публичные импорты используют существующие родительские пакеты проекта. Их зависимости, включая `esmpy` для сеточной интерполяции, не изменены; примеры выполняются в полном рабочем окружении проекта. Изолированные тесты адаптеров не означают, что необязательные зависимости родительских пакетов больше не нужны.
