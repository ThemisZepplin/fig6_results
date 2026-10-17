# Cell 5 — physical-size plotting/QC, no formal acceptance.
capture(5)

COLORS = {
    'ERA5': 'black',
    'Hot17': 'firebrick',
    'Middle17': '#1f77b4',
    'Cold17': 'seagreen',
}

LABELS = {
    'NCHN_tp': 'TP (mm day⁻¹)',
    'z500_anom_NCHN': 'Z500 anomaly (gpm)',
    'sm_avg': 'SM (m³ m⁻³)',
    'SHF_Avg': 'SHF (W m⁻²)',
    'NCVI_concurrent': 'NCVI anomaly (PVU)',
    'TCC_Avg': 'Total cloud cover (%)',
    'partial_fitted_Tmax': 'SM–SHF diagnostic (°C)',
}

STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': None,
}

MARKERS = {
    'ERA5': 'o',
    'Hot17': '^',
    'Middle17': 's',
    'Cold17': 'v',
}

LINESTYLES = {
    'ERA5': '-',
    'Hot17': '-',
    'Middle17': '--',
    'Cold17': ':',
}


def limits(values):
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]

    if not len(array):
        raise ValueError('No finite values for panel range')

    lo = float(array.min())
    hi = float(array.max())
    pad = max((hi - lo) * 0.10, abs(lo) * 0.01, 0.001)

    locator = MaxNLocator(
        nbins=4,
        steps=[1, 2, 2.5, 5, 10],
    )
    ticks = locator.tick_values(lo - pad, hi + pad)

    return (float(ticks[0]), float(ticks[-1])), ticks


def legend(fig, reference=True, position='bottom'):
    names = ('ERA5', *GROUPS) if reference else GROUPS
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[group],
            marker=MARKERS[group],
            linestyle=LINESTYLES[group],
            lw=1.4,
            label=group,
        )
        for group in names
    ]

    if position == 'top':
        location = 'upper center'
        anchor = (0.5, 0.935)
    elif position == 'bottom':
        location = 'lower center'
        anchor = (0.5, 0.005)
    else:
        raise ValueError('Legend position must be top or bottom')

    fig.legend(
        handles=handles,
        loc=location,
        ncol=len(names),
        frameon=False,
        bbox_to_anchor=anchor,
        fontsize=8,
        borderaxespad=0,
    )


def style(ax):
    ax.grid(
        True,
        linestyle=':',
        alpha=0.2,
        linewidth=0.4,
    )
    ax.tick_params(width=0.7, length=3)

    for spine in ax.spines.values():
        spine.set_color('0.5')
        spine.set_linewidth(0.6)


def display_units(df):
    """Return a plotting copy; never rescale source/member/model CSV values."""

    result = df.copy(deep=True)
    audit = []
    tcc = result.factor.eq('TCC_Avg')

    source_masks = [
        ('ERA5', tcc & result.group.eq('ERA5')),
        ('S2S', tcc & result.group.ne('ERA5')),
    ]

    for origin, mask in source_masks:
        if not mask.any():
            continue

        values = result.loc[mask, 'value'].to_numpy(float)

        if (
            not np.isfinite(values).all()
            or (0 > values).any()
            or (values > 100).any()
        ):
            raise ValueError(
                f'{origin} Total cloud cover outside [0,100]; '
                'verify source units'
            )

        unit = TCC_SOURCE_UNITS.get(origin, 'auto')
        evidence = 'explicit_configuration'

        if unit == 'auto':
            # Values no greater than 1 are interpreted as fractions.
            unit = 'fraction' if 1 >= values.max() else 'percent'
            evidence = (
                'range_inference; percentage samples no greater than 1 '
                'require explicit override'
            )

        if unit not in ('fraction', 'percent'):
            raise ValueError('TCC unit must be auto/fraction/percent')

        if unit == 'fraction' and values.max() > 1:
            raise ValueError(
                f'{origin} TCC declared fraction but exceeds 1'
            )

        multiplier = 100.0 if unit == 'fraction' else 1.0

        for column in ('value', *BAND_COLUMNS):
            if column in result:
                result.loc[mask, column] = (
                    result.loc[mask, column] * multiplier
                )

        audit.append(
            dict(
                source=origin,
                input_unit=unit,
                display_unit='percent',
                multiplier=multiplier,
                evidence=evidence,
            )
        )

    return result, audit


def band_values(frame):
    lower, upper = BAND_COLUMNS
    present = [column in frame for column in BAND_COLUMNS]

    if any(present) and not all(present):
        raise ValueError('Both uncertainty bounds must be supplied')

    if not all(present):
        return None

    lo = frame[lower].to_numpy(float)
    hi = frame[upper].to_numpy(float)

    if np.isnan(lo).all() and np.isnan(hi).all():
        return None

    if (
        not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or (lo > hi).any()
    ):
        raise ValueError('Invalid uncertainty bounds')

    values = frame.value.to_numpy(float)

    if (lo > values).any() or (values > hi).any():
        raise ValueError(
            'Uncertainty bounds do not contain central values'
        )

    return lo, hi


def row_limits(frame, factor, window, init=None):
    values = frame.value.to_numpy(float)

    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonfinite/empty row data')

    envelopes = [values]

    # Check bands per curve so curves without bands remain valid.
    for _, group_frame in frame.groupby(['init_date', 'group']):
        band = band_values(group_frame)
        if band is not None:
            envelopes.extend(band)

    envelope = np.concatenate(envelopes)

    total_preferred = {
        'z500_anom_NCHN': (-50.0, 100.0, 25.0),
        'NCHN_tp': (-1.0, 10.0, 1.0),
        'NCVI_concurrent': (-0.30, 0.25, 0.10),
        'TCC_Avg': (0.0, 100.0, 25.0),
    }

    stage_preferred = {
        ('2023-06-05', 'partial_fitted_Tmax'): (28.0, 32.0, 1.0),
        ('2023-06-08', 'partial_fitted_Tmax'): (32.0, 35.0, 0.5),
        ('2023-06-05', 'z500_anom_NCHN'): (25.0, 125.0, 25.0),
        ('2023-06-05', 'SHF_Avg'): (50.0, 80.0, 10.0),
        ('2023-06-05', 'NCVI_concurrent'): (-0.40, 0.10, 0.10),
        (None, 'TCC_Avg'): (0.0, 100.0, 25.0),
    }

    if window == 'Total':
        preferred = total_preferred.get(factor)
    else:
        preferred = stage_preferred.get(
            (init, factor),
            stage_preferred.get((None, factor)),
        )

    if preferred is None:
        ylim, ticks = limits(envelope)
        base = None
    else:
        lo, hi, step = preferred
        base = (lo, hi)

        if factor == 'TCC_Avg':
            if (0 > envelope).any() or (envelope > 100).any():
                raise ValueError(
                    'TCC bands exceed physical percent bounds'
                )

        # Expand outward on the requested grid if a curve or band exceeds it.
        if lo > envelope.min():
            lo = float(np.floor(envelope.min() / step) * step)

        if envelope.max() > hi:
            hi = float(np.ceil(envelope.max() / step) * step)

        ticks = np.round(
            np.arange(lo, hi + step * 0.5, step),
            10,
        )

        if factor == 'NCHN_tp' and envelope.min() >= 0:
            ticks = ticks[ticks >= 0]

        ylim = (lo, hi)

    info = dict(
        factor=factor,
        preferred_limits=base,
        actual_limits=ylim,
        curve_and_band_min=float(envelope.min()),
        curve_and_band_max=float(envelope.max()),
        expanded=(base is not None and tuple(ylim) != base),
        bands_checked=True,
    )

    return ylim, ticks, info


def timeseries_figure(
    df,
    factors,
    window,
    version=None,
    inits=None,
):
    inits = tuple(INITS if inits is None else inits)

    if len(inits) != 2:
        raise ValueError(
            'Process figure requires exactly two initialization columns'
        )

    rows = len(factors)
    data = df if version is None else df[df.model_version == version]
    data = data[data.init_date.isin(inits)].copy()
    data, unit_audit = display_units(data)

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(6.5, 8.9),
        dpi=600,
    )

    end = '2023-06-24' if window == 'Total' else '2023-06-18'
    dates = pd.date_range('2023-06-14', end, freq='D')

    for row, factor in enumerate(factors):
        row_data = data[data.factor == factor]

        for col, init in enumerate(inits):
            ax = axes[row, col]
            series = row_data[row_data.init_date == init].copy()
            series['date'] = pd.to_datetime(series.date)

            ylim, ticks, range_info = row_limits(
                series,
                factor,
                window,
                init=init,
            )
            ax._process_range_qc = dict(
                range_info,
                init_date=init,
            )

            for group in ('ERA5', *GROUPS):
                group_series = series[
                    series.group == group
                ].sort_values('date')

                if group_series.empty and group == 'ERA5':
                    if factor == 'NCVI_concurrent':
                        raise ValueError(
                            'Mandatory ERA5 NCVI reference missing: '
                            f'{init}/{window}'
                        )
                    continue

                aligned = np.array_equal(
                    pd.DatetimeIndex(group_series.date).values,
                    dates.values,
                )
                if len(group_series) != len(dates) or not aligned:
                    raise ValueError(
                        'Incomplete or misaligned daily plot series: '
                        f'{init}/{factor}/{group}'
                    )

                if not np.isfinite(
                    group_series.value.to_numpy(float)
                ).all():
                    raise ValueError(
                        'Nonfinite plot series: '
                        f'{init}/{factor}/{group}'
                    )

                band = band_values(group_series)
                if band is not None:
                    ax.fill_between(
                        group_series.date,
                        *band,
                        color=COLORS[group],
                        alpha=0.12,
                        linewidth=0,
                    )

                ax.plot(
                    group_series.date,
                    group_series.value,
                    color=COLORS[group],
                    marker=MARKERS[group],
                    linestyle=LINESTYLES[group],
                    markersize=4.2,
                    markeredgewidth=0.5,
                    markeredgecolor=COLORS[group],
                    lw=1.3,
                )

            ax.set_ylim(ylim)
            ax.set_yticks(ticks)

            if factor == 'NCVI_concurrent':
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda value, pos: (
                            f'{0.0 if 1e-9 > abs(value) else value:.2f}'
                        )
                    )
                )

            ax.set_xlim(
                dates[0] - pd.Timedelta(hours=12),
                dates[-1] + pd.Timedelta(hours=12),
            )
            ax.set_xticks(dates)
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%m-%d')
            )

            if row == rows - 1:
                angle = 70 if window == 'Total' else 0
                ax.tick_params(
                    axis='x',
                    labelbottom=True,
                    labelsize=8 if window == 'Total' else 9,
                )
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=angle,
                    ha='right' if angle else 'center',
                    rotation_mode='anchor',
                )
            else:
                ax.tick_params(axis='x', labelbottom=False)

            if window == 'Stage-I':
                ax.axvline(
                    pd.Timestamp('2023-06-18'),
                    color='0.4',
                    linestyle='--',
                    lw=0.8,
                )
                ax.axvspan(
                    pd.Timestamp('2023-06-18'),
                    pd.Timestamp('2023-06-19'),
                    color='0.9',
                    alpha=0.5,
                    zorder=-1,
                )

            ax.set_title(
                f'({chr(97 + row * 2 + col)})',
                loc='left',
            )
            ax.set_ylabel(LABELS[factor] if col == 0 else '')
            style(ax)

            if window == 'Total':
                ax.tick_params(axis='y', labelsize=8)

    for col, init in enumerate(inits):
        fig.text(
            0.345 if col == 0 else 0.79,
            0.965,
            pd.Timestamp(init).strftime('%d June %Y'),
            ha='center',
            fontsize=10,
            fontweight='semibold',
        )

    legend(
        fig,
        reference=bool((data.group == 'ERA5').any()),
        position='top',
    )

    # Keep the original panel height after moving the shared legend upward.
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        top=0.88,
        bottom=0.085 if window == 'Total' else 0.06,
        wspace=0.24,
        hspace=0.23 if window == 'Total' else 0.43,
    )

    # Thin numeric tick labels only when an expanded range would overlap.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax in axes.ravel():
        labels = ax.get_yticklabels()
        stride = 1

        if ax._process_range_qc['expanded']:
            while len(labels) > stride:
                boxes = [
                    label.get_window_extent(renderer)
                    for index, label in enumerate(labels)
                    if index % stride == 0
                ]
                separated = all(
                    1 >= (
                        min(first.y1, second.y1)
                        - max(first.y0, second.y0)
                    )
                    for first, second in zip(boxes[:-1], boxes[1:])
                )
                if separated:
                    break
                stride += 1

            for index, label in enumerate(labels):
                label.set_visible(index % stride == 0)

        ax._process_range_qc['tick_label_stride'] = stride

    fig._tcc_unit_audit = unit_audit

    return fig, axes.ravel().tolist()


def pdf_page_qc(page, expected_height_inches):
    """Measure exported PDF size, editable text, and readable font sizes."""

    spans = [
        span
        for block in page.get_text('dict')['blocks']
        if 'lines' in block
        for line in block['lines']
        for span in line['spans']
        if span['text'].strip()
    ]
    body = [
        span
        for span in spans
        if not (span.get('flags', 0) & 1)
    ]
    scripts = [
        span
        for span in spans
        if span.get('flags', 0) & 1
    ]
    reasons = []

    width_mm = page.rect.width / 72 * 25.4
    height_mm = page.rect.height / 72 * 25.4

    if width_mm > 170.01 or height_mm > 228.01:
        reasons.append('AGU physical size limit exceeded')

    if (
        abs(page.rect.width - 6.5 * 72) > 0.1
        or abs(
            page.rect.height - expected_height_inches * 72
        ) > 0.1
    ):
        reasons.append('Export canvas differs from intended size')

    if not body:
        reasons.append('No editable ordinary text')

    if any(7.99 > span['size'] for span in body):
        reasons.append('Ordinary label below 8pt')

    if any(5.49 > span['size'] for span in scripts):
        reasons.append('Superscript/subscript below 5.5pt')

    text_clipped = any(
        -0.1 > span['bbox'][0]
        or -0.1 > span['bbox'][1]
        or span['bbox'][2] > page.rect.width + 0.1
        or span['bbox'][3] > page.rect.height + 0.1
        for span in spans
    )
    if text_clipped:
        reasons.append('Actual PDF text clipped')

    return dict(
        status='FAIL' if reasons else 'PASS',
        reasons='; '.join(reasons),
        width_mm=width_mm,
        height_mm=height_mm,
        min_body_pt=min(
            (span['size'] for span in body),
            default=None,
        ),
        min_script_pt=min(
            (span['size'] for span in scripts),
            default=None,
        ),
        body_spans=len(body),
        script_spans=len(scripts),
        editable_text_spans=len(spans),
        body_minimum_pt=8,
        script_minimum_pt=5.5,
    )


def figure_qc(fig, axes):
    if (
        fig.get_figwidth() * 25.4 > 170
        or fig.get_figheight() * 25.4 > 228
    ):
        raise ValueError('AGU physical dimensions exceeded')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sizes = np.array(
        [
            [ax.get_position().width, ax.get_position().height]
            for ax in axes
        ]
    )

    if not np.allclose(sizes, sizes[0], rtol=0, atol=1e-10):
        raise ValueError('Unequal process-panel axes boxes')

    daily_ticks_aligned = None
    total_dates_all_labelled = None

    if len(axes) in (10, 12):
        window_end = (
            '2023-06-24'
            if len(axes) == 10
            else '2023-06-18'
        )
        expected_dates = pd.date_range(
            '2023-06-14',
            window_end,
            freq='D',
        )
        expected_x = mdates.date2num(
            expected_dates.to_pydatetime()
        )

        for ax in axes:
            if not np.allclose(
                ax.get_xticks(),
                expected_x,
                atol=1e-9,
                rtol=0,
            ):
                raise ValueError(
                    'Daily tick locations do not match calendar-day nodes'
                )

            member_lines = [
                line
                for line in ax.lines
                if line.get_marker() in set(MARKERS.values())
            ]
            if len(member_lines) not in (3, 4):
                raise ValueError(
                    'Expected three S2S group curves and optional ERA5'
                )

            for line in member_lines:
                line_dates = pd.DatetimeIndex(
                    pd.to_datetime(line.get_xdata(orig=True))
                )
                if not np.array_equal(
                    line_dates.values,
                    expected_dates.values,
                ):
                    raise ValueError(
                        'Time-series marker lies between daily tick nodes'
                    )

        daily_ticks_aligned = True

        if len(axes) == 10:
            required_labels = [
                date.strftime('%m-%d')
                for date in expected_dates
            ]

            for ax in axes[-2:]:
                labels = ax.get_xticklabels()

                if [label.get_text() for label in labels] != required_labels:
                    raise ValueError(
                        'Fig.5 does not label every day, '
                        '06-14 through 06-24'
                    )

                rotations_valid = all(
                    tick.get_rotation() >= 65
                    and 75 >= tick.get_rotation()
                    for tick in labels
                )
                if not rotations_valid:
                    raise ValueError(
                        'Fig.5 daily tick labels require angled display'
                    )

                spans = [
                    tick.get_window_extent(renderer)
                    for tick in labels
                ]
                if any(
                    first.overlaps(second)
                    for first, second in zip(spans[:-1], spans[1:])
                ):
                    raise ValueError(
                        'Fig.5 angled daily labels overlap'
                    )

            total_dates_all_labelled = True

    if len(axes) == 4:
        # Validate the asymmetric Fig. S11 annotation layout.
        for index, ax in enumerate(axes):
            boxes = [
                text
                for text in ax.texts
                if 'n = 51' in text.get_text()
            ]
            if len(boxes) != 1:
                raise ValueError(
                    'Fig.S11 must show exactly one statistics box per panel'
                )

            expected_alignment = (
                'left'
                if ax.get_xlabel().startswith('SM–SHF')
                else 'right'
            )
            if boxes[0].get_ha() != expected_alignment:
                raise ValueError(
                    'Fig.S11 statistics box misplaced '
                    f'in panel {index}'
                )

    for ax in axes:
        info = getattr(ax, '_process_range_qc', None)
        if info is not None:
            lo, hi = ax.get_ylim()
            if (
                lo - 1e-10 > info['curve_and_band_min']
                or info['curve_and_band_max'] > hi + 1e-10
            ):
                raise ValueError(
                    'Curve/uncertainty band clipped by y limits'
                )

    texts = list(fig.texts)
    for ax in axes:
        texts += [ax._left_title, ax.xaxis.label, ax.yaxis.label]
        texts += list(ax.texts)
        texts += ax.get_xticklabels()
        texts += ax.get_yticklabels()

    boxes = [
        (text.get_text(), text.get_window_extent(renderer))
        for text in texts
        if text.get_visible() and text.get_text()
    ]

    for label, box in boxes:
        if (
            0 > box.x0
            or 0 > box.y0
            or box.x1 > fig.bbox.width
            or box.y1 > fig.bbox.height
        ):
            raise ValueError(
                f'Clipped text in process figure: {label}'
            )

    for index, (label, first_box) in enumerate(boxes):
        for other, second_box in boxes[index + 1:]:
            horizontal_overlap = (
                min(first_box.x1, second_box.x1)
                - max(first_box.x0, second_box.x0)
            )
            vertical_overlap = (
                min(first_box.y1, second_box.y1)
                - max(first_box.y0, second_box.y0)
            )
            if horizontal_overlap > 1 and vertical_overlap > 1:
                raise ValueError(
                    'Overlapping process labels: '
                    f'{label}; {other}'
                )

    return {
        'equal_boxes': True,
        'text_extent_qc': 'PASS',
        'daily_marker_tick_alignment': daily_ticks_aligned,
        'fig5_all_11_days_labelled': total_dates_all_labelled,
        'width_inches': fig.get_figwidth(),
        'height_inches': fig.get_figheight(),
        'axes_count': len(axes),
        'curve_and_band_coverage': (
            'PASS'
            if any(
                hasattr(ax, '_process_range_qc')
                for ax in axes
            )
            else None
        ),
    }


def scatter_statistics(df, version, init, factor):
    members = df[
        (df.init_date == init)
        & (df.model_version == version)
    ].sort_values('member')

    if (
        len(members) != 51
        or not np.array_equal(
            members.member.to_numpy(),
            np.arange(51),
        )
    ):
        raise ValueError(
            'Descriptive scatter statistics require 51 original members'
        )

    x = members[factor].to_numpy(float)
    y = members.y_tmax.to_numpy(float)

    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        raise ValueError(
            'Pearson/trend undefined for nonfinite or constant data'
        )

    dx = x - x.mean()
    dy = y - y.mean()
    r = float(
        np.dot(dx, dy)
        / np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    )
    slope = float(np.dot(dx, dy) / np.dot(dx, dx))
    intercept = float(y.mean() - slope * x.mean())

    return dict(
        init_date=init,
        model_version=version,
        factor=factor,
        n=51,
        pearson_r=r,
        descriptive_R2=r * r,
        trend_slope=slope,
        trend_intercept=intercept,
        R2_definition=(
            'Squared pairwise Pearson correlation; '
            'not complete MLR R2'
        ),
    )


def scatter_figure(
    df,
    version,
    inits=None,
    factors_by_init=None,
):
    inits = tuple(INITS if inits is None else inits)

    if len(inits) != 2:
        raise ValueError(
            'Scatter figure requires two initialization columns'
        )

    factors_by_init = (
        SCATTER_FACTORS_BY_INIT
        if factors_by_init is None
        else factors_by_init
    )

    invalid_factor_layout = (
        set(factors_by_init) != set(inits)
        or any(
            len(factors_by_init[init]) != 2
            for init in inits
        )
    )
    if invalid_factor_layout:
        raise ValueError(
            'Scatter figure requires two factors '
            'for each displayed initialization'
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.5, 5.9),
        dpi=600,
    )
    ylim, yticks = limits(
        df[df.init_date.isin(inits)].y_tmax
    )

    for col, init in enumerate(inits):
        for row, factor in enumerate(factors_by_init[init]):
            ax = axes[row, col]
            group_data = df[
                (df.init_date == init)
                & (df.model_version == version)
            ]

            for group in GROUPS:
                group_members = group_data[
                    group_data.group == group
                ]
                ax.scatter(
                    group_members[factor],
                    group_members.y_tmax,
                    s=23,
                    marker=MARKERS[group],
                    color=COLORS[group],
                    alpha=0.7,
                    edgecolor='white',
                    linewidth=0.3,
                )

            xlim, xticks = limits(group_data[factor])
            stats = scatter_statistics(
                df,
                version,
                init,
                factor,
            )
            xx = np.array(xlim)
            ax.plot(
                xx,
                stats['trend_intercept']
                + stats['trend_slope'] * xx,
                color='0.3',
                lw=1.0,
                ls='--',
                zorder=1,
            )

            # Partial-fitted statistics stay on the left.
            if factor != 'partial_fitted_Tmax':
                stat_x = 0.975
                stat_alignment = 'right'
            else:
                stat_x = 0.025
                stat_alignment = 'left'

            ax.text(
                stat_x,
                0.965,
                (
                    f"r = {stats['pearson_r']:.2f}\n"
                    f"R² = r² = {stats['descriptive_R2']:.2f}\n"
                    'n = 51'
                ),
                transform=ax.transAxes,
                ha=stat_alignment,
                va='top',
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.88,
                    pad=2.0,
                ),
            )

            ax.set_xlim(xlim)
            ax.set_xticks(xticks)
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
            ax.set_xlabel(LABELS[factor])
            ax.set_ylabel('6.14–17 Tmax (°C)')
            ax.set_title(
                f'({chr(97 + row * 2 + col)})',
                loc='left',
            )
            style(ax)

    for col, init in enumerate(inits):
        fig.text(
            0.33 if col == 0 else 0.79,
            0.960,
            pd.Timestamp(init).strftime('%d June %Y'),
            ha='center',
            fontsize=10,
            fontweight='semibold',
        )

    legend(fig, reference=False)
    fig.subplots_adjust(
        left=0.14,
        right=0.955,
        top=0.94,
        bottom=0.14,
        wspace=0.40,
        hspace=0.50,
    )

    return fig, axes.ravel().tolist()


# Read and verify dedicated plotting CSV files before drawing.
for name, expected_hash in plot_hashes.items():
    if sha256(OUTPUT / name) != expected_hash:
        raise ValueError('Plot CSV changed')


fig5_plot = pd.read_csv(
    OUTPUT / 'fig5_total_timeseries_plot.csv'
)

fig11_plot = pd.concat(
    [
        pd.read_csv(
            OUTPUT
            / (
                'fig11_stageI_timeseries_'
                + version
                + '_plot.csv'
            )
        )
        for version in VERSIONS
    ],
    ignore_index=True,
)

scatter_plot = pd.concat(
    [
        pd.read_csv(
            OUTPUT
            / (
                'figS11_stageI_scatter_'
                + version
                + '.csv'
            )
        )
        for version in VERSIONS
    ],
    ignore_index=True,
)


built = []
layout_qc = []
axis_range_qc = []
tcc_unit_qc = []


with plt.rc_context(STYLE):
    fig, axes = timeseries_figure(
        fig5_plot,
        FACTORS5,
        'Total',
        inits=TOTAL_INITS,
    )
    built.append(
        ('Fig5_Total_DIAGNOSTIC', fig, axes)
    )

    for pair in PROCESS_PAIRS:
        tag = '_'.join(
            pd.Timestamp(init).strftime('%m%d')
            for init in pair
        )

        for version in VERSIONS:
            fig, axes = timeseries_figure(
                fig11_plot,
                ('partial_fitted_Tmax', *FACTORS11),
                'Stage-I',
                version,
                inits=pair,
            )
            built.append(
                (
                    f'Fig11_June14_17_{tag}_'
                    f'{version}_DIAGNOSTIC',
                    fig,
                    axes,
                )
            )

            fig, axes = scatter_figure(
                scatter_plot,
                version,
                inits=pair,
                factors_by_init=SCATTER_FACTORS_BY_INIT,
            )
            built.append(
                (
                    f'FigS11_June14_17_{tag}_mechanism_'
                    f'{version}_DIAGNOSTIC',
                    fig,
                    axes,
                )
            )

    for name, fig, axes in built:
        layout_qc.append(
            dict(
                filename=name,
                **figure_qc(fig, axes),
            )
        )
        axis_range_qc.extend(
            dict(
                filename=name,
                **ax._process_range_qc,
            )
            for ax in axes
            if hasattr(ax, '_process_range_qc')
        )
        tcc_unit_qc.extend(
            dict(filename=name, **item)
            for item in getattr(fig, '_tcc_unit_audit', [])
        )
        display(fig)


scatter_stats = pd.DataFrame(
    [
        scatter_statistics(
            scatter_plot,
            version,
            init,
            factor,
        )
        for version in VERSIONS
        for init in FIG11_INITS
        for factor in SCATTER_FACTORS_BY_INIT[init]
    ]
)


print(
    'Cell 5 layout QC PASS; residualized only; '
    'all figures DIAGNOSTIC, '
    'PNG600 dpi / vector PDF+SVG'
)
