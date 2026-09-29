# Визуализация буевых результатов

`libs.validation.buoy_visualization` строит графики непосредственно из
`validator.results`, собранных `RawFieldAggregator`. Допускается также словарь,
полученный через `pickle.load` из файла `Validator.save`.

```python
from libs.validation.buoy_visualization import (
    BuoyPanel,
    plot_buoy_scatter,
    plot_buoy_track,
    plot_buoy_metric_map,
)

results = sst_b_validator.results
source = "nemo5_sst_hindcast_on_buoy_observations"
reference = "buoy_observations"
```

Функции возвращают фигуру и оси Matplotlib, не вызывают `show()` и не сохраняют
файлы. Например, `fig.savefig("comparison.png", dpi=200, bbox_inches="tight")`.
Единицы берутся из метаданных; преобразования единиц внутри отрисовки нет.

## 1. Scatter: источник по X, эталон по Y

```python
fig, ax = plot_buoy_scatter(
    results, source, reference,
    variable="sst",
    start="2023-01-01", end="2023-12-31",
    show_ci=True,                 # по умолчанию False
    show_bin_means=True,          # по умолчанию False
    mean_bin_width=0.5,
    min_points_per_bin=20,
    hist_bins=100,
)
```

Цвет показывает число наблюдений в двумерном бине. N, Bias, RMSE, MAE и
корреляция рассчитываются по всем конечным сопоставленным парам. Bias всегда
означает **источник минус эталон**. Отрицательные температуры и выбросы
не отбрасываются. `xlim`, `ylim`, `xymin` и `xymax` меняют только отображение:
N и статистика остаются прежними. Неопределённая корреляция отображается как N/A.

Доверительные интервалы — существующие приближённые 95% интервалы для Bias
(нормальная аппроксимация) и RMSE (дельта-метод). Они предполагают независимость
отсчётов и не учитывают временную зависимость измерений одного буя.

Для скорости и направления дрейфа:

```python
uv = ("drift_eastward", "drift_northward")
drift_source = "nemo5_drift_hindcast_on_buoy_observations"

fig, ax = plot_buoy_scatter(
    drift_results, drift_source, reference,
    variable=uv, reduction="norm",
)
fig, ax = plot_buoy_scatter(
    drift_results, drift_source, reference,
    variable=uv, reduction="direction",
    circular_view="nearest", show_bin_means=True,
)
```

Компоненты указываются явно в порядке восток, север. `direction` использует
`atan2(northward, eastward)`: 0° на восток, 90° на север. Нулевые векторы
исключаются из направлений; скорость нулевого вектора остаётся допустимой.

`circular_view="wrapped"` (по умолчанию) отображает обе оси от 0 до 360°.
`"nearest"` сдвигает эталон на целый период к ближайшему эквивалентному значению:
пара X=359°, Y=1° отображается как X=359°, Y=361°. Это не добавляет наблюдений.
Bias, RMSE и MAE используют кратчайшую угловую разность; корреляция и средние
по бинам — циклические. Угловой остаток принадлежит `[-period/2, period/2)`.

Для уже сохранённой скалярной угловой переменной укажите `circular=True`;
`period=360` для градусов или `period=2*np.pi` для радиан. Вычисление
`reduction="direction"` всегда возвращает градусы и требует `period=360`.

## 2. Временные панели одного буя и карта трека

```python
fig, axes = plot_buoy_track(
    results, buoy_id="<ID из field.meta['bids']>", reference=reference,
    sources=[source],
    panels=[
        BuoyPanel("identity", variable="sst", label="SST"),
        BuoyPanel("difference", variable="sst", label="Bias"),
        BuoyPanel("mae", variable="sst", label="Absolute error"),
    ],
    start="2023-01-01", end="2023-12-31",
    show_summary=True,           # по умолчанию False
    max_gap="6h",               # необязательно
    source_labels={source: "NEMO", reference: "Buoys"},
    source_styles={source: {"color": "tab:blue"}},
)
axes["panels"][0].set_ylim(-3, 12)
```

`axes` содержит `panels` (список временных осей), `map` и `summary`
(ось таблицы либо None). Панель `identity` включает источник и эталон;
остальные берут готовые метрики из точного ключа `(source, reference)`.
Если есть только обратная пара, функция объясняет ошибку: знак сохранённой
метрики автоматически не меняется.

`sources` может содержать несколько источников. Цвет каждого сохраняется
между панелями. Для дрейфа из `nemo5_validation.ipynb`:

```python
fig, axes = plot_buoy_track(
    drift_results, buoy_id="<ID>", reference=reference,
    sources=[drift_source],
    panels=[
        BuoyPanel("identity", uv, "norm", "Drift speed"),
        BuoyPanel("difference→norm", label="Vector error"),
        BuoyPanel("angle_error", label="Angle error"),
    ],
    show_summary=True,
)
```

Названия метрик должны совпадать с ключами результатов; составные метрики
используют символ `→`. Для уже сведённой к скаляру метрики `(buoy,time)`
не нужно задавать переменную или повторное `reduction`.

Линии разрываются на недействительных отсчётах и временных пропусках больше
`max_gap`. По умолчанию порог равен 1,5 медианного положительного шага
query time. Изолированные позиции сохраняются на карте в виде маркеров;
перекрывающиеся окна с повторяющимися моментами рисуются отдельными кривыми.
Таблица усредняет все показанные отсчёты каждой серии, без выравнивания масок
между источниками; для панелей `reduction="direction"` среднее циклическое.

## 3. Все ошибки на одной карте

```python
fig, ax = plot_buoy_metric_map(
    results, "difference", (source, reference),
    variable="sst",
    start="2023-06-01", end="2023-08-31",
    s=10, alpha=0.7,
    map_kwargs={"central_longitude": 120, "coastline_resolution": "110m"},
    title="SST error, summer 2023",
)
```

Каждый валидный отсчёт — отдельный маркер в своей локации. Пространственного
или временного усреднения нет. Точный порядок `dataset_key` сохраняется.
Для значений обоих знаков автоматически выбирается расходящаяся шкала с
центром в нуле, для неотрицательных — последовательная. Настройки `cmap`,
`norm` или `vmin`/`vmax` позволяют переопределить шкалу (`norm` не совмещается
с `vmin`/`vmax`).

Карты используют `visualization.create_cartopy_axes`, `visualize_scatter`
и `visualize_trajectory`. По умолчанию используется их северная полярная
проекция. Можно передать `proj`, существующую `ax`, либо `extent` в порядке
`[lon_min, lon_max, lat_min, lat_max]`. Автоматические границы рассчитываются
по точкам в проекции карты, включая треки через 180° долготы.

Для проверки без скачивания береговой линии:

```python
map_kwargs = dict(add_land=False, add_coastlines=False, add_gridlines=False)
```

## Отбор и сохранение метаданных

- `start`/`end` применяются к `field.meta["datetimes"]`, а не только к дате окна
  агрегатора. Дата без времени включает целые сутки, datetime задаёт точную
  границу; обе границы включаются. Время интерпретируется как UTC.
- `buoy_ids=[...]` ограничивает scatter и карту выбранными буями. Используйте
  полные идентификаторы из метаданных, включая префикс источника, если он есть.
- Соответствие scatter проверяется по окну валидации, ID и времени, с проверкой
  координат и единиц. Используется пересечение отсчётов; обрезания по длине нет.
- Каждое окно остаётся отдельной группой наблюдений. Если окна перекрываются,
  повторные оценки участвуют в N и средних отдельно. Координаты трека в один
  момент времени объединяются только при их согласованности.
- Новые pickle-файлы сохраняют метаданные обеих реализаций `MetricField`.
  Старые файлы по-прежнему читаются, но отсутствующие метаданные не появляются.

В `notebooks/res_buoy_2023-01-01_2023-12-31.pkl` метаданные уже потеряны.
Если исходный валидатор ещё находится в памяти сервера, сначала загрузите
исправленную реализацию класса, убедитесь, что `.meta` заполнено, затем
выполните сохранение под новым именем:

```python
field = next(iter(
    sst_b_validator.results["identity"][(reference,)]["RawFieldAggregator"].values()
))
assert {"bids", "coords", "datetimes", "dims", "valid", "units"} <= field.meta.keys()
sst_b_validator.save("res_buoy_2023_with_metadata.pkl")
```

При обновлении работающего ноутбука его autoreload должен применить новые
методы к `MetricField`; после сохранения проверьте `.meta` загруженного поля.
Если исходных результатов в памяти больше нет, потребуется повторный расчёт.
Повторное сохранение загруженного старого файла не восстанавливает метаданные.
