import os 
import numpy as np
import matplotlib.pyplot as plt
import libs.validation.visualization as visualization 
from libs.validation.metrics import VectorNorm
from libs.validation.aggregators import SpatialAggregator, GlobalTemporalAggregator, RegionalTemporalAggregator, SeasonalSpatialAggregator, AverageAggregator


interesting_areas = [20, 79, 102, 107, 112, 122, 138, 139, 141, 143, 144, 145, 146, 151, 153, 156, 179, 180, 182, 188, 218, 219, 220, 240]

def plot_vector_map_comparison(grid, results, metric_name='difference', variable='Sea Ice Drift', units='cm/s', step=16, filename=None, ds_names=None, colormap='viridis'):
    names = [ds for ds in results[metric_name]] if ds_names is None else ds_names
    fig, axes = visualization.create_cartopy_grid(len(names), 3, coastline_resolution='110m', ax_size=12)
    last_mappable = None
    samples = {ds[0]: SpatialAggregator().finalize(results['identity'][ds]['SpatialAggregator']) for ds in results['identity']}
    comparisons = [SpatialAggregator().finalize(results[metric_name][name]['SpatialAggregator']) for name in names]
    for i, comparison in enumerate(comparisons):
        vmin = min([np.nanpercentile(VectorNorm()(samples[name]), 1) for name in names[i]])
        vmax = max([np.nanpercentile(VectorNorm()(samples[name]), 99) for name in names[i]])
        scale = vmax * 10 * 3
        for j, name in enumerate(names[i]):
            sample = samples[(name)]
            params = visualization.get_color_params(colormap, vmin, vmax)
            axes[i, j].set_title(f'{name}')
            # Plot and store the mappable
            last_mappable = visualization.visualize_full_vector_field(
                axes[i, j], grid,
                sample,
                from_polar=False, from_direction=False, scale=scale,step=step,
                **params)
            axes[i, j].text(
                0.99, 0.01, f'max: {np.nanmax(VectorNorm()(sample)):.2f}\nmin: {np.nanmin(VectorNorm()(sample)):.2f}',
                transform=axes[i, j].transAxes,       # uses fraction of the axes
                ha='right', va='bottom',       # align text relative to the point
                fontsize=14,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                            )
        fig.colorbar(
            last_mappable,
            ax=axes[i,:-1].tolist(),
            location='right',
            aspect=20,
            pad=0.02,
            label=f'{variable}, ({units})',
            extend='both',)
        vmin = np.nanpercentile(VectorNorm()(comparison), 1)
        vmax = np.nanpercentile(VectorNorm()(comparison), 99)
        scale = vmax * 10 * 6
        params = visualization.get_color_params('viridis', vmin, vmax)
        axes[i, -1].set_title(f"{' vs '.join(names[i])}")
        # Plot and store the mappable
        last_mappable = visualization.visualize_full_vector_field(
            axes[i, -1], grid,
            comparison,
            from_polar=False, from_direction=False, scale=scale,step=step,
            **params
        )
        axes[i, -1].text(
            0.91, 0.01, f'max: {np.nanmax(VectorNorm()(comparison)):.2f}\nmin: {np.nanmin(VectorNorm()(comparison)):.2f}',
            transform=axes[i, -1].transAxes,       # uses fraction of the axes
            ha='left', va='bottom',       # align text relative to the point
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
        fig.colorbar(
            last_mappable,
            ax=axes[i, -1],
            location='right',
            aspect=20,
            pad=0.02,
            label=f'{variable}, ({units})',
            extend='both',
        ) 
        fig.suptitle(f'Sea Ice Drift', fontsize='x-large', y=1.02)
        if filename is not None:
            fig.savefig(filename)
    plt.show()

def plot_season_vector_map_comparison(grid, results, metric_name='difference', variable='Sea Ice Drift', units='cm/s', step=16, filename=None, ds_names=None, colormap='viridis'):
    names = [ds for ds in results[metric_name]] if ds_names is None else ds_names
    seasons = list(results[metric_name][names[0]]['SeasonalSpatialAggregator'].keys())
    for s, season in enumerate(seasons):
        fig, axes = visualization.create_cartopy_grid(len(names), 3, coastline_resolution='110m', ax_size=12)
        last_mappable = None
        samples = {ds[0]: SeasonalSpatialAggregator().finalize(results['identity'][ds]['SeasonalSpatialAggregator'])[season] for ds in results['identity']}
        comparisons = [SeasonalSpatialAggregator().finalize(results[metric_name][name]['SeasonalSpatialAggregator'])[season] for name in names]
        for i, comparison in enumerate(comparisons):
            vmin = min([np.nanpercentile(VectorNorm()(samples[name]), 1) for name in names[i]])
            vmax = max([np.nanpercentile(VectorNorm()(samples[name]), 99) for name in names[i]])
            scale = vmax * 10 * 3
            for j, name in enumerate(names[i]):
                sample = samples[(name)]
                params = visualization.get_color_params(colormap, vmin, vmax)
                axes[i, j].set_title(f'{name}')
                # Plot and store the mappable
                last_mappable = visualization.visualize_full_vector_field(
                    axes[i, j], grid,
                    sample,
                    from_polar=False, from_direction=False, scale=scale,step=step,
                    **params)
                axes[i, j].text(
                    0.99, 0.01, f'max: {np.nanmax(VectorNorm()(sample)):.2f}\nmin: {np.nanmin(VectorNorm()(sample)):.2f}',
                    transform=axes[i, j].transAxes,       # uses fraction of the axes
                    ha='right', va='bottom',       # align text relative to the point
                    fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                                )
            fig.colorbar(
                last_mappable,
                ax=axes[i,:-1].tolist(),
                location='right',
                aspect=20,
                pad=0.02,
                label=f'{variable}, ({units})',
                extend='both',)
            vmin = np.nanpercentile(VectorNorm()(comparison), 1)
            vmax = np.nanpercentile(VectorNorm()(comparison), 99)
            scale = vmax * 10 * 6
            params = visualization.get_color_params('viridis', vmin, vmax)
            axes[i, -1].set_title(f"{' vs '.join(names[i])}")
            # Plot and store the mappable
            last_mappable = visualization.visualize_full_vector_field(
                axes[i, -1], grid,
                comparison,
                from_polar=False, from_direction=False, scale=scale,step=step,
                **params
            )
            axes[i, -1].text(
                0.91, 0.01, f'max: {np.nanmax(VectorNorm()(comparison)):.2f}\nmin: {np.nanmin(VectorNorm()(comparison)):.2f}',
                transform=axes[i, -1].transAxes,       # uses fraction of the axes
                ha='left', va='bottom',       # align text relative to the point
                fontsize=14,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )
            fig.colorbar(
                last_mappable,
                ax=axes[i, -1],
                location='right',
                aspect=20,
                pad=0.02,
                label=f'{variable}, ({units})',
                extend='both',
            ) 
        fig.suptitle(f'Sea Ice Drift during {season} season', fontsize='x-large', y=1.02)
        if filename is not None:
            fig.savefig(filename)
    plt.show()


def plot_err_vector_map(grid, norm, angle, name, variable='Sea Ice Drift', units='cm/s', filename=None):
    fig, ax = visualization.create_cartopy(coastline_resolution='50m')
    # Store the last mappable for colorbar creation

    vmin, vmax = np.nanpercentile(norm, 1), np.nanpercentile(norm, 99)
    params = visualization.get_color_params('viridis', vmin, vmax)
    ax.set_title(f'{name}')
    # Plot and store the mappable
    last_mappable = visualization.visualize_full_vector_field(
        ax, grid,
        (norm, angle),
        from_polar=True, from_direction=False,
        **params
    )
    fig.colorbar(
        last_mappable,
        ax=ax,
        location='right',
        aspect=20,
        pad=0.02,
        label=f'{variable}, ({units})',
        extend='both',
    )
    if filename is not None:
        fig.savefig(filename)
    plt.show()

def plot_err_vector_map_grid(grid, norms, angles, names, variable='Sea Ice Frift', units='cm/s', filename=None, aggregator_name='SpatialAggregator'):
    fig, axes = visualization.create_cartopy_grid(len(norms), 1, coastline_resolution='50m')
    # Store the last mappable for colorbar creation
    last_mappable = None
    for i, (norm, angle) in enumerate(zip(norms, angles)):
        vmin, vmax = np.nanpercentile(norm, 1), np.nanpercentile(norm, 99)
        params = visualization.get_color_params('viridis', vmin, vmax)
        axes[i, 0].set_title(f'{names[i]}')
        # Plot and store the mappable
        last_mappable = visualization.visualize_full_vector_field(
            axes[i, 0], grid,
            (norm, angle),
            from_polar=True, from_direction=False,
            **params
        )
        fig.colorbar(
            last_mappable,
            ax=axes[i,:].tolist(),
            location='right',
            aspect=20,
            pad=0.02,
            label=f'{variable}, ({units})',
            extend='both',
        )
    fig.suptitle(f'metrics', fontsize='x-large', y=1.02)
    if filename is not None:
        fig.savefig(filename)
    plt.show()

def global_balance_evolution(results, ax=None, variable='Sea Ice concentration (SIC)', filepath=None, datasets=None, colors=None, **params):
    datasets = list(results['identity'].keys()) if datasets is None else datasets
    cmap = plt.colormaps['Set1']  # Use 'tab20' for up to 20 colors
    colors = cmap(np.linspace(0, 1, 9)) if colors is None else colors
    for i, dataset in enumerate(datasets):
            fig, ax = visualization.plot_error_evolution(results['identity'][dataset]['GlobalTemporalAggregator'],
                                                       ax=ax,
                                                       color=colors[i],
                                                       label=f"{' vs '.join(datasets[i], )}", 
                                                       title=f"Arctic {variable} balance evolution",
                                                       ylabel=f'region mean {variable}')
    if filepath is not None:
        fig.savefig(os.path.join(filepath, f"{'_'.join(variable.lower().split(' '))}_monthly_balance_global.jpg"))
    return fig, ax


def areal_balance_evolution(results, areas_lookup, areas, balance_key='identity', variable='Sea Ice concentration (SIC)', units='%', filepath=None, **params):
    datasets = list(results[balance_key].keys())
    areas = list(results[balance_key][datasets[0]]['RegionalTemporalAggregator'].keys()) if areas is None else areas
    areas_names = [areas_lookup[area]['properties']['SUB_REGION'] for area in areas]
    cmap = plt.colormaps['Set1']  # Use 'tab20' for up to 20 colors
    colors = cmap(np.linspace(0, 1, 9))
    for area_i, area in enumerate(areas):
        fig, ax = visualization.plot_error_evolution(
            results[balance_key][datasets[0]]['RegionalTemporalAggregator'][area],
            color=colors[0],
            label=f"{' vs '.join(datasets[0], )}",
            title=f"{variable} balance evolution at {areas_names[area_i]}",
            ylabel=f'region mean {variable}, {units}',
        )
        for i, dataset in enumerate(datasets[1:]):
                visualization.plot_error_evolution(results[balance_key][dataset]['RegionalTemporalAggregator'][area],
                                                   ax=ax,
                                                   color=colors[i+1],
                                                   label=f"{' vs '.join(datasets[i+1], )}", 
                                                   title=f"{variable} balance evolution at {areas_names[area_i]}",
                                                   ylabel=f'region mean {variable}, {units}')
        if filepath is not None:
            fig.savefig(os.path.join(filepath, f"{'_'.join(variable.lower().split(' '))}_monthly_balance_{areas_names[area_i]}.jpg"))
    return areas_names

def global_evolution_over(results, over, variable='Sea Ice Thickness', units='m', filepath=None, aggregator='GlobalTemporalAggregator', title=None, **params):
    cmap = plt.colormaps['Set1']  # Use 'tab20' for up to 20 colors
    colors = cmap(np.linspace(0, 1, 9))
    datasets = list(results['identity'].keys())
    fig, ax = visualization.plot_error_evolution(
        results['identity'][(over,)][aggregator],  
        color=colors[0],
        marker=None,
        label=f'{over}',
        title=title or f"Arctic {variable} balance",
        ylabel=f'mean {variable}, {units}',
        **params
    )
    datasets.remove((over,))
    print(datasets)
    for i, dataset in enumerate(datasets):
        visualization.plot_error_evolution(results['identity_over'][(dataset[0], over)][aggregator],
                                       ax=ax,
                                       color=colors[i+1],
                                       label=f"{dataset[0]}", 
                                       title=title or f"Arctic {variable} balance",
                                       ylabel=f'mean {variable}, {units}',
                                       **params)
    ax.legend()
    if filepath is not None:
            fig.savefig(os.path.join(filepath, f"{'_'.join(variable.lower().split(' '))}_monthly_balance_global.jpg"))
    plt.show()

def global_balance(results, variable='Sea Ice Concentration', units='%', filepath=None, aggregator='GlobalTemporalAggregator', **params):
    cmap = plt.colormaps['Set1']  # Use 'tab20' for up to 20 colors
    colors = cmap(np.linspace(0, 1, 9))
    datasets = list(results['identity'].keys())
    fig, ax = visualization.plot_error_cycle(
        results['identity'][datasets[0]][aggregator],  
        color=colors[0],
        marker=None,
        label=f'{datasets[0]} median',
        cycle='daily',
        aggregation_func='nanmedian',
        **params
    )
    for i, dataset in enumerate(datasets[1:]):
                visualization.plot_error_cycle(results['identity'][dataset][aggregator],
                                               ax=ax,
                                               color=colors[i+1],
                                               label=f"{' vs '.join(datasets[i+1], )} median", 
                                               cycle='daily',
                                               title=f"Arctic {variable} balance",
                                               ylabel=f'mean {variable}, {units}',
                                               **params)
    ax.legend()
    if filepath is not None:
            fig.savefig(os.path.join(filepath, f"{'_'.join(variable.lower().split(' '))}_monthly_balance_global.jpg"))
    plt.show()

def global_balance_over(results, over, variable='Sea Ice Thickness', units='m', filepath=None, aggregator='GlobalTemporalAggregator', **params):
    cmap = plt.colormaps['Set1']  # Use 'tab20' for up to 20 colors
    colors = cmap(np.linspace(0, 1, 9))
    datasets = list(results['identity'].keys())
    fig, ax = visualization.plot_error_cycle(
        results['identity'][(over,)][aggregator],  
        color=colors[0],
        marker=None,
        label=f'{over}',
        cycle='daily',
        aggregation_func='nanmedian',
        **params
    )
    datasets.remove((over,))
    print(datasets)
    for i, dataset in enumerate(datasets):
        visualization.plot_error_cycle(results['identity_over'][(dataset[0], over)][aggregator],
                                       ax=ax,
                                       color=colors[i+1],
                                       label=f"{dataset[0]}", 
                                       cycle='daily',
                                       title=f"Arctic {variable} balance",
                                       ylabel=f'mean {variable}, {units}',
                                       **params)
    ax.legend()
    if filepath is not None:
            fig.savefig(os.path.join(filepath, f"{'_'.join(variable.lower().split(' '))}_monthly_balance_global.jpg"))
    plt.show()

def areal_balance(results, areas_lookup, areas, variable='Sea Ice concentration (SIC)', units='%', filepath=None, **params):
    datasets = list(results['identity'].keys())
    areas = list(results['identity'][datasets[0]]['RegionalTemporalAggregator'].keys()) if areas is None else areas
    areas_names = [areas_lookup[area]['properties']['SUB_REGION'] for area in areas]
    cmap = plt.colormaps['Set1']  # Use 'tab20' for up to 20 colors
    colors = cmap(np.linspace(0, 1, 9))
    for area_i, area in enumerate(areas):
        fig, ax = visualization.plot_error_cycle(
            results['identity'][datasets[0]]['RegionalTemporalAggregator'][area],
            color=colors[0],
            label=f"{' vs '.join(datasets[0], )} median",
            cycle='daily',
            aggregation_func='nanmedian',
            title=f"{variable} balance at {areas_names[area_i]}",
            # interdecile_range=True,
            **params
        )
        for i, dataset in enumerate(datasets[1:]):
                visualization.plot_error_cycle(results['identity'][dataset]['RegionalTemporalAggregator'][area],
                                               ax=ax,
                                               color=colors[i+1],
                                               label=f"{' vs '.join(datasets[i+1], )} median", 
                                               # interdecile_range=True, 
                                               cycle='daily',
                                               title=f"{variable} balance at {areas_names[area_i]}",
                                               ylabel=f'region mean {variable}, {units}',
                                               **params)
        if filepath is not None:
            fig.savefig(os.path.join(filepath, f"{'_'.join(variable.lower().split(' '))}_monthly_balance_{areas_names[area_i]}.jpg"))
    return areas_names

def plot_err_map_grid(results, grid, variable='Sea Ice concentration (SIC), %', suptitle=None, filename=None, metrics=None, ds_names=None, **kwargs):
    aggregator_name = 'SpatialAggregator'
    if metrics is None:
        metrics = list(results.keys()) if metrics is None else metrics
        metrics = [metric for metric in metrics if len(list(results[metric].keys())[0]) >= 2]
    ds_names = list(results[metrics[0]].keys()) if ds_names is None else ds_names
    fig, axes = visualization.create_cartopy_grid(len(metrics), len(ds_names), coastline_resolution='50m')
    # Store the last mappable for colorbar creation
    last_mappable = None
    for i, metric in enumerate(metrics):
        vmin = np.nanmin([np.nanpercentile(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name]), 1) for ds_name in ds_names])
        vmax = np.nanmax([np.nanpercentile(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name]), 99) for ds_name in ds_names])
        params = visualization.get_color_params(metric, vmin, vmax)
        for j, ds_name in enumerate(ds_names):
            ax_i, ax_j = i, j
            axes[ax_i, ax_j].set_title(f'{" vs ".join(ds_name, )}: {metric} metric')
            
            # Plot and store the mappable
            last_mappable = visualization.visualize_scalar_field(
                axes[ax_i, ax_j], grid, 
                np.squeeze(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name])), 
                **{**params, **kwargs}
            )
            axes[i, j].text(
                0.99, 0.01, f'mean: {AverageAggregator().finalize(results[metric][ds_name][aggregator_name]):.2f}',
                transform=axes[i, j].transAxes,       # uses fraction of the axes
                ha='right', va='bottom',       # align text relative to the point
                fontsize=14,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        if isinstance(variable, (list, tuple)):
            cur_variable = variable[i]
        else:
            cur_variable = variable
        fig.colorbar(
            last_mappable, 
            ax=axes[i,:].tolist(),  # All axes in this row
            location='right',
            aspect=20,
            pad=0.02,
            label=f'{cur_variable}',
            extend='both'
        )
    if suptitle is not None:
        fig.suptitle(f'{suptitle}', fontsize='x-large', y=1.02)
    if filename is not None:
        fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    return fig, axes


def plot_seasonal_err_map_grid(
    results, grid, target_dataset, metric,
    variable='Sea Ice concentration (SIC), %',
    figsize=None, suptitle=None, filename=None, ds_names=None,
    axes_names=None, proj=None, seasons=None, **kwargs,
):
    """Plot one metric against one target, with datasets in rows and seasons in columns.

    `ds_names` contains tested dataset names (not result-key tuples). If omitted,
    all pairwise comparisons involving `target_dataset` are selected in result
    order. Stored pair order and metric signs are preserved. `seasons` defaults
    to all seasons present in the selected results, in their stored order.

    All panels share a color scale based on their 1st/99th percentiles, unless
    `vmin`, `vmax`, or `norm` is supplied. Other keyword arguments, including
    `lat` and `lon` for `grid=None`, reach `visualize_scalar_field`.
    Missing seasons are labeled "No data". Return `(fig, axes)`, where `axes`
    is always a two-dimensional array, including for a single panel.

    This replaces the former multi-metric, one-figure-per-season interface.
    """
    aggregator_name = 'SeasonalSpatialAggregator'
    if metric not in results:
        raise ValueError(f"Metric {metric!r} is not present in results.")
    metric_results = results[metric]
    comparisons = {}
    for pair in metric_results:
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        if target_dataset not in pair or pair[0] == pair[1]:
            continue
        dataset = pair[1] if pair[0] == target_dataset else pair[0]
        comparisons.setdefault(dataset, []).append(pair)

    if ds_names is None:
        ds_names = list(comparisons)
    elif isinstance(ds_names, str):
        ds_names = [ds_names]
    else:
        ds_names = list(ds_names)
    if not ds_names:
        raise ValueError(f"No datasets selected for comparison with {target_dataset!r}.")

    selected_pairs = []
    seasonal_maps = []
    for dataset in ds_names:
        pairs = comparisons.get(dataset, [])
        if not pairs:
            raise ValueError(
                f"No pairwise results for {dataset!r} vs {target_dataset!r} "
                f"and metric {metric!r}."
            )
        if len(pairs) != 1:
            raise ValueError(
                f"Ambiguous comparison for {dataset!r} vs {target_dataset!r}: {pairs!r}."
            )
        pair = pairs[0]
        if aggregator_name not in metric_results[pair]:
            raise ValueError(f"Missing {aggregator_name} for comparison {pair!r}.")
        selected_pairs.append(pair)
        seasonal_maps.append(
            SeasonalSpatialAggregator.finalize(metric_results[pair][aggregator_name])
        )

    available_seasons = list(dict.fromkeys(
        season for maps in seasonal_maps for season in maps
    ))
    if seasons is None:
        seasons = available_seasons
    elif isinstance(seasons, str):
        seasons = [seasons]
    else:
        seasons = list(seasons)
    if not seasons:
        raise ValueError("No seasons selected or available in the seasonal results.")
    unknown_seasons = [season for season in seasons if season not in available_seasons]
    if unknown_seasons:
        raise ValueError(f"Seasons not present in the selected results: {unknown_seasons!r}.")

    nrows, ncols = len(ds_names), len(seasons)
    if axes_names is not None and (
        len(axes_names) != nrows or any(len(row) != ncols for row in axes_names)
    ):
        raise ValueError(f"axes_names must have shape ({nrows}, {ncols}).")

    # Finalize each comparison once and derive limits only from selected panels.
    bounds = []
    for maps in seasonal_maps:
        for season in seasons:
            if season in maps:
                field = np.asarray(maps[season])
                finite = field[np.isfinite(field)]
                if finite.size:
                    bounds.append(np.percentile(finite, [1, 99]))
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    auto_min, auto_max = vmin is None, vmax is None
    if auto_min:
        vmin = min(bound[0] for bound in bounds) if bounds else 0.0
    if auto_max:
        vmax = max(bound[1] for bound in bounds) if bounds else 1.0
    if vmin == vmax:
        # A nonzero span also keeps TwoSlopeNorm valid for all-zero differences.
        padding = abs(vmin) * 0.01 or 1.0
        if auto_min:
            vmin -= padding
        if auto_max:
            vmax += padding

    params = {**visualization.get_color_params(metric, vmin, vmax), **kwargs}
    if 'norm' in kwargs:
        params.update(vmin=vmin, vmax=vmax)
    norm = params.get('norm')
    if norm is not None and not isinstance(norm, str):
        if norm.vmin is None:
            norm.vmin = vmin
        if norm.vmax is None:
            norm.vmax = vmax
        params.pop('vmin', None)
        params.pop('vmax', None)

    fig, axes = visualization.create_cartopy_axes(
        nrows, ncols, coastline_resolution='50m', proj=proj, figsize=figsize,
    )
    axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)
    last_mappable = None
    for i, (pair, maps) in enumerate(zip(selected_pairs, seasonal_maps)):
        for j, season in enumerate(seasons):
            ax = axes[i, j]
            title = (
                axes_names[i][j] if axes_names is not None
                else f'{" vs ".join(pair)}: {metric}, {season}'
            )
            ax.set_title(title)
            if season not in maps:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                continue
            last_mappable = visualization.visualize_scalar_field(
                ax, grid, maps[season], **params,
            )

    fig.colorbar(
        last_mappable, ax=axes.ravel().tolist(), location='right',
        aspect=40, pad=0.02, label=variable, extend='both',
    )
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize='x-large', y=1.02)
    if filename is not None:
        fig.savefig(filename, dpi=600, bbox_inches='tight')
    return fig, axes

def plot_err_map_comparison(results, grid, variable='Sea Ice concentration (SIC), %', suptitle=None, filename=None, metrics=None, ds_names=None, **kwargs):
    aggregator_name = 'SpatialAggregator'
    if metrics is None:
        metrics = list(results.keys()) if metrics is None else metrics
        metrics = [metric for metric in metrics if len(list(results[metric].keys())[0]) >= 2]
    ds_names = list(results[metrics[0]].keys()) if ds_names is None else ds_names
    fig, axes = visualization.create_cartopy_grid(len(metrics), len(ds_names), coastline_resolution='50m')
    # Store the last mappable for colorbar creation
    last_mappable = None
    for i, metric in enumerate(metrics):
        vmin = np.nanmin([np.nanpercentile(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name]), 1) for ds_name in ds_names])
        vmax = np.nanmax([np.nanpercentile(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name]), 99) for ds_name in ds_names])
        params = visualization.get_color_params(metric, vmin, vmax)
        for j, ds_name in enumerate(ds_names):
            axes[i, j].set_title(f'{" vs ".join(ds_name, )}: {metric} metric')
            
            # Plot and store the mappable
            last_mappable = visualization.visualize_scalar_field(
                axes[i, j], grid, 
                np.squeeze(SpatialAggregator().finalize(results['identity'][ds_name][aggregator_name])), 
                **{**params, **kwargs}
            )
            axes[i, j].text(
                0.99, 0.01, f"mean: {AverageAggregator().finalize(results['identity'][ds_name][aggregator_name]):.2f}",
                transform=axes[i, j].transAxes,       # uses fraction of the axes
                ha='right', va='bottom',       # align text relative to the point
                fontsize=14,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        if isinstance(variable, (list, tuple)):
            cur_variable = variable[i]
        else:
            cur_variable = variable
        fig.colorbar(
            last_mappable, 
            ax=axes[i,:].tolist(),  # All axes in this row
            location='right',
            aspect=20,
            pad=0.02,
            label=f'{cur_variable}',
            extend='both'
        )

        axes[i, -1].set_title(f'{" vs ".join(ds_names, )}: {metric} metric')
        
        # Plot and store the mappable
        last_mappable = visualization.visualize_scalar_field(
            axes[i, -1], grid, 
            np.squeeze(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name])), 
            **{**params, **kwargs}
        )
    if suptitle is not None:
        fig.suptitle(f'{suptitle}', fontsize='x-large', y=1.02)
    if filename is not None:
        fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    return fig, axes

def plot_err_map_table(results, grid, column_datasets, row_datasets, metric, variable='Sea Ice concentration (SIC), %', figsize=None, suptitle=None, filename=None, axes_names=None, proj=None, **kwargs):
    aggregator_name = 'SpatialAggregator'

    ds_names = list(results[metric].keys())
    fig, axes = visualization.create_cartopy_axes(len(row_datasets), len(column_datasets), coastline_resolution='50m', proj=proj, figsize=figsize)
    axes = np.asarray(axes, dtype=object).reshape(
        len(row_datasets), len(column_datasets)
    )
    # Store the last mappable for colorbar creation
    last_mappable = None
    
    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = np.nanmin([np.nanpercentile(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name]), 1) for ds_name in ds_names])
    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = np.nanmax([np.nanpercentile(SpatialAggregator().finalize(results[metric][ds_name][aggregator_name]), 99) for ds_name in ds_names])
    
    params = visualization.get_color_params(metric, vmin, vmax)
    for i, row_ds in enumerate(row_datasets):
        for j, column_ds in enumerate(column_datasets):
            # if any((row_ds, column_ds) == ds[:2] for ds in ds_names):
            #     ds_pair = (row_ds, column_ds)
            # else:
            #     ds_pair = (column_ds, row_ds)
            ds_combo_name = next(
                (ds for ds in ds_names if ds[:2] in [(row_ds, column_ds), (column_ds, row_ds)]),
                None
            )
            if axes_names is not None:
                axes[i, j].set_title(axes_names[i][j])
            else:    
                axes[i, j].set_title(f'{" vs ".join(ds_combo_name, )}: {metric}')
            
            # Plot and store the mappable
            last_mappable = visualization.visualize_scalar_field(
                axes[i, j], grid, 
                np.squeeze(SpatialAggregator().finalize(results[metric][ds_combo_name][aggregator_name])), 
                **{**params, **kwargs}
            )
            axes[i, j].text(
                0.99, 0.01, f'mean: {AverageAggregator().finalize(results[metric][ds_combo_name][aggregator_name]):.2f}',
                transform=axes[i, j].transAxes,       # uses fraction of the axes
                ha='right', va='bottom',       # align text relative to the point
                fontsize=14,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    fig.colorbar(
        last_mappable, 
        ax=axes[:,:],  # All axes in this row
        location='right',
        aspect=40,
        pad=0.02,
        label=f'{variable}',
        extend='both'
    )
    if suptitle is not None:
        fig.suptitle(f'{suptitle}', fontsize='x-large', y=1.02)
    if filename is not None:
        fig.savefig(filename, dpi=600, bbox_inches='tight')
    return fig, axes
