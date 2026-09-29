import numpy as np
import xarray as xr
from pathlib import Path
from torch.utils.data import Dataset


def lon_to_180_range(lon):
    return (lon + 180) % 360 - 180

class SentinelSIEDataset(Dataset):
    """
    Dataset размеченных снимков Sentinel, выдающий поле Sea Ice Extent (SIE)
    за запрошенную дату.

    Ожидаем структура:
        root/
            2025-11-01/fields.nc
            2025-11-02/fields.nc
            ...

    B каждом fields.nc должны быть переменные:
        water : 1 где нет льда, 0 иначе
        ice_1 : 1 где есть лёд, 0 иначе

    Логика SIE:
        SIE = 1  -> есть лёд  (ice_1 == 1)
        SIE = 0  -> нет льда (water == 1 и ice_1 != 1)
        SIE = nan -> всё остальное (неопределённо / суша / облака / конфликт)
    """

    def __init__(
            self,
            data_folder,
            dst_grid = None,
            ice_var: str = "ice_1",
            water_var: str = "water",
            add_coords: bool = False,
            name: str = None,
    ):
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.path = Path(data_folder)
        self.ice_var = ice_var
        self.water_var = water_var
        self.add_coords = add_coords

        # Собираем словарь {дата(D): путь к файлу}
        self.dates_dict = self._create_dates_dict()
        self.dates = np.array(sorted(self.dates_dict.keys()))

        # Создаём "сетку" (долгота/широта) из первого файла, если возможно
        self.dst_grid = dst_grid        
        self.grid = self.dst_grid

    def _create_dates_dict(self):
        """
        Ищем все files.nc внутри подкаталогов, где имя директории — дата YYYY-MM-DD.
        """
        result = {}
        files = sorted(self.path.glob("*/fields.nc"))
        for file in files:
            try:
                # Имя директории — дата
                date = np.datetime64(file.parent.name)
                date = date.astype("datetime64[D]")
            except ValueError:
                print(f"skipping {file} (cannot parse date from folder name)")
                continue
            if date in result:
                print(f"warning: duplicate sentinel file for date {date}, using first one")
                continue
            result[date] = file
        print(f"parsed {len(result)} Sentinel SIE dates")
        return result

    def _create_grid(self):
        """
        Пытаемся прочитать широту/долготу из одного из файлов.
        Если координат нет, используем индексы пикселей как "псевдо-координаты".
        """
        if not self.dates_dict:
            raise RuntimeError("No Sentinel files found for SentinelSIEDataset")

        sample_file = next(iter(self.dates_dict.values()))
        print(f"loading Sentinel grid from {sample_file}")
        with xr.open_dataset(sample_file, cache=False) as ds:
            # Попробуем стандартные имена координат
            if "latitude" in ds.coords and "longitude" in ds.coords:
                lat = ds["latitude"].values
                lon = ds["longitude"].values
                lon = lon_to_180_range(lon)

                # Если координаты 1D — разворачиваем в 2D сетку
                if lat.ndim == 1 and lon.ndim == 1:
                    lon, lat = np.meshgrid(lon, lat)
            else:
                # Фоллбэк: координат нет -> используем индексы
                field = ds[self.water_var]
                h, w = field.shape[-2], field.shape[-1]
                y = np.arange(h)
                x = np.arange(w)
                lon, lat = np.meshgrid(x, y)
            grid = {"longitude": lon, "latitude": lat}
        return grid

    def _load_sie_for_day(self, day: np.datetime64) -> np.ndarray:
        """
        Читает файл за конкретный день и строит поле SIE (2D массив).
        """
        day = np.datetime64(day).astype("datetime64[D]")
        if day not in self.dates_dict:
            print(f"No Sentinel SIE file for date {day}")
            return None
        file = self.dates_dict[day]
        with xr.open_dataset(file, cache=False) as ds:
            if (self.water_var in ds) and (self.ice_var in ds):
                water = ds[self.water_var].values.astype(np.float32)
                ice = ds[self.ice_var].values.astype(np.float32)
            else:
                print('No ice/water var in ds!')
                return None

        # SIE: 1 — лёд, 0 — нет льда, nan — остальное
        sie = np.full_like(ice, np.nan, dtype=np.float32)

        ice_mask = (ice == 1)
        # вода только там, где явно нет льда (приоритет льда в случае конфликта)
        water_mask = (water == 1) & ~ice_mask

        sie[ice_mask] = 1.0
        sie[water_mask] = 0.0

        return sie  # shape: (H, W)

    def __len__(self):
        # Чисто формальная длина (кол-во доступных дней)
        return len(self.dates)

    def __getitem__(self, date, length=None, add_coords=None):
        """
        date может быть:
        - np.datetime64
        - строка c датой
        - целое число (индекс по self.dates)

        Возвращает:
            np.ndarray формы (seq_len, C, H, W),
            где C = 1 (канал SIE) [+ 2 координатных канала, если add_coords=True].
        """
        # Приводим индекс к дате
        if isinstance(date, (int, np.integer)):
            # трактуем как индекс по self.dates
            if date < 0 or date >= len(self.dates):
                raise IndexError(f"index {date} out of range for SentinelSIEDataset")
            day = self.dates[date]
        else:
            # строка / datetime64
            if not isinstance(date, np.datetime64):
                date = np.datetime64(date)
            # нас интересует только день (снимок на день)
            day = date.astype("datetime64[D]")

        sie = self._load_sie_for_day(day)   # (H, W)
        return sie[None]
