import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
import cartopy.crs as ccrs
from libs.validation import visualization


def prepare_diagnostic_data(region_temps, forcing_temps_dict,
                            depths=None, vmin=None, vmax=None):

    times = sorted(region_temps.keys())
    tnums = mdates.date2num(times)
    dt_half = (np.diff(tnums).mean() / 2.0) if len(times) > 1 else 0.5
    time_edges = np.concatenate([tnums - dt_half, [tnums[-1] + dt_half]])
    
    # Океанские темпы
    T = np.vstack([region_temps[t] for t in times])
    n_t, n_z = T.shape

    # Глубины и их границы
    if depths is None:
        depths = np.arange(n_z)
    else:
        depths = np.array(depths)
    dz = (np.diff(depths).mean() if n_z > 1 else 1.0)
    depth_edges = np.concatenate([depths - dz/2.0, [depths[-1] + dz/2.0]])
    
    # Атмосферные знач.
    forcing_arrays = [
        np.array([fdict.get(t, np.array([np.nan]))[0] - 273.15 for t in times])
        for fdict in forcing_temps_dict.values()
    ]
    
    # Общие vmin/vmax
    all_forc = np.hstack(forcing_arrays)
    if vmin is None:
        vmin = min(np.nanmin(T), np.nanmin(all_forc))
    if vmax is None:
        vmax = max(np.nanmax(T), np.nanmax(all_forc))
    
    # Нормировка
    if vmin < 0 < vmax:
        max_abs = max(abs(vmin), abs(vmax))
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    return {
        'times': times,
        'time_edges': time_edges,
        'depths': depths,
        'depth_edges': depth_edges,
        'T': T,
        'forcing_arrays': forcing_arrays,
        'forcing_names': list(forcing_temps_dict.keys()),
        'vmin': vmin,
        'vmax': vmax,
        'norm': norm
    }

def draw_forcings(fig, left_spec, data, start_row=0, cmap='viridis'):
    axes = []
    for i, name in enumerate(data['forcing_names']):
        row = len(fig.axes)
        spec = left_spec[row+start_row, 0]
        ax = fig.add_subplot(spec)
        mat = data['forcing_arrays'][i].reshape(1, -1)
        ax.pcolormesh(data['time_edges'], [0,1], mat,
                      cmap=cmap, norm=data['norm'], shading='flat')
        ax.set_yticks([0.5])
        ax.set_yticklabels([name])
        ax.set_xticks([])
        axes.append(ax)
    return axes

def draw_ocean(fig, left_spec, data, row=-1, cmap='viridis'):
    spec = left_spec[row, 0]
    ax = fig.add_subplot(spec)
    pcm = ax.pcolormesh(data['time_edges'],
                        data['depth_edges'],
                        data['T'].T,
                        cmap=cmap,
                        norm=data['norm'],
                        shading='flat')
    ax.set_ylabel('Depth, levels')
    ax.set_xlabel('Date')
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    return ax, pcm

def draw_mask(fig, right_spec, region_mask, grid, region_name):
    _, ax = visualization.create_cartopy(fig=fig, ax=right_spec)
    visualization.visualize_scalar_field(ax, grid,
                           np.where(region_mask == region_name, 1, np.nan))
    ax.axis('off')
    return ax

def draw_sea_ice(fig, left_spec, sea_ice_data, data, ocean_ax, row=0):
    """
    Рисует эволюцию Sea Ice над ocean-осью, разделяя с ней ось X полностью.
    
    Параметры:
      fig           — Figure, в котором мы уже нарисовали ocean-ось
      left_spec     — GridSpec для левой колонки
      sea_ice_data  — dict source_name -> dict(date -> scalar)
      data          — результат prepare_diagnostic_data (нужен для times)
    """
    if sea_ice_data is None:
        return []
    spec = left_spec[row, 0]
    # 3) создать новую ось sharex=ocean_ax
    ax = fig.add_subplot(spec, sharex=ocean_ax)
    # 4) нарисовать линии для каждого источника точно по data['times']
    for name, d in sea_ice_data.items():
        y = [np.asarray(d.get(t, np.nan)).item() for t in data['times']]
        ax.plot(data['times'], y, marker='o', linestyle='-', label=name)
    # 5) оформление
    ax.set_ylabel("Sea ice area, $km^2$")
    ax.legend(fontsize="small")
    # убрать подписи X — они на ocean_ax
    plt.setp(ax.get_xticklabels(), visible=False)

    return [ax]

def draw_fluxes(fig, left_spec, fluxes_data, data, ocean_ax, row=0, y_label="Heat fluxes, $Wt/m^2$"):
    """
    Рисует эволюцию Sea Ice над ocean-осью, разделяя с ней ось X полностью.
    
    Параметры:
      fig           — Figure, в котором мы уже нарисовали ocean-ось
      left_spec     — GridSpec для левой колонки
      fluxes_data   — dict source_name -> dict(date -> scalar)
      data          — результат prepare_diagnostic_data (нужен для times)
    """
    if fluxes_data is None:
        return []
    spec = left_spec[row, 0]
    # 3) создать новую ось sharex=ocean_ax
    ax = fig.add_subplot(spec, sharex=ocean_ax)
    # 4) нарисовать линии для каждого источника точно по data['times']
    for name, d in fluxes_data.items():
        y = [np.asarray(d.get(t, np.nan)).item() for t in data['times']]
        ax.plot(data['times'], y, marker='o', linestyle='-', label=name)
    # 5) оформление
    ax.set_ylabel(y_label)
    ax.legend(fontsize="small")
    # убрать подписи X — они на ocean_ax
    plt.setp(ax.get_xticklabels(), visible=False)

    return [ax]
    
def plot_region_diagnostic(region_temps,
                           forcing_temps_dict,
                           depths=None,
                           region=None,
                           region_name='',
                           cmap='viridis',
                           vmin=None, vmax=None,
                           ice_data=None,
                           flux_data=None,
                           albedo_data=None,
                           region_mask=None, grid=None,
                           mask_proj=ccrs.PlateCarree()):
    """
    Модульная функция для diagnostic plot: temperature, forcings и маска.
    """
    data = prepare_diagnostic_data(region_temps, forcing_temps_dict,
                                   depths, vmin, vmax)
    
    # Фигура и разметка
    has_ice = ice_data is not None
    has_mask = region_mask is not None
    has_fluxes = flux_data is not None
    has_albedo = albedo_data is not None
    n_f = len(data['forcing_names'])
    fig = plt.figure(figsize=(12+2*has_mask, 4 + n_f+ 3*has_ice + 3*has_fluxes+3*has_albedo))
    outer = fig.add_gridspec(1, 1+has_mask, width_ratios=[4]+ has_mask*[1], wspace=0.2)
    left_spec = outer[0].subgridspec(has_albedo + has_fluxes + has_ice+n_f+1, 1,
                                     height_ratios=[3]*has_albedo + [3]*has_fluxes + [3]*has_ice + [1]*n_f + [4],
                                     hspace=0.02)
    right_spec = outer[1] if has_mask else None

    # right_spec = outer[1]

    # Рисуем forcings и ocean
    axes = []
    # Sea ice (если есть)
    
    # Forcings
    axes += draw_forcings(fig, left_spec, data, start_row=has_ice+has_fluxes+has_albedo, cmap=cmap)
    # Ocean
    ocean_ax, pcm = draw_ocean(fig, left_spec, data, cmap=cmap)
    axes.append(ocean_ax)
    if has_ice:
        axes = draw_sea_ice(fig, left_spec, ice_data, data, ocean_ax=ocean_ax, row=has_fluxes+has_albedo) + axes
    if has_fluxes:
        axes = draw_fluxes(fig, left_spec, flux_data, data, ocean_ax=ocean_ax, row=has_albedo) + axes
    if has_albedo:
        axes = draw_fluxes(fig, left_spec, albedo_data, data, ocean_ax=ocean_ax, row=0, y_label = "Albedo") + axes
    # Общий colorbar
    fig.colorbar(pcm, ax=axes,
                 orientation='vertical',
                 fraction=0.04,
                 pad=0.02).set_label('Temperature, C')

    # Маска справа (если нужна)
    if has_mask:
        axm = draw_mask(fig, right_spec, region_mask, grid, region)
        axes.append(axm)

    fig.suptitle(region_name, y=0.94)
    return fig, axes
