# Utilities shared by the analysis scripts

TEXT_WIDTH = 7.3  # inches, matching \mytextwidth (528pt) in the document, so figures drawn at this width are not rescaled

def paired_by_seed(res1, res2, column):
    """Values of `column` from results tables `res1` and `res2`, paired by seed

    Raises an error unless both tables contain the same seeds once each.
    """
    vals1 = res1.set_index('seed')[column].sort_index()
    vals2 = res2.set_index('seed')[column].sort_index()
    if not (vals1.index.is_unique and vals1.index.equals(vals2.index)):
        raise ValueError(f'Seeds do not match: {vals1.index.tolist()} vs {vals2.index.tolist()}')
    return vals1.to_numpy(), vals2.to_numpy()

def set_log_ticks(ax):
    """Give a log-scale y-axis a few readable ticks, with no minor ticks

    Call this after plotting, as it fixes the axis limits.
    Uses ticks at 1, 2 and 5 times powers of 10 if there are at least 3 in range,
    then tries 1, 2, 3 and 5 times,
    and otherwise (for a narrow range) uses evenly spaced ticks.
    Labels are plain numbers when all ticks lie in [0.001, 10000), and otherwise
    in scientific notation.
    """
    import numpy as np
    from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator, NullLocator

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi)
    powers = 10. ** np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
    for multiples in [[1, 2, 5], [1, 2, 3, 5]]:
        ticks = np.outer(powers, multiples).ravel()
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        if len(ticks) >= 3:
            break
    if len(ticks) > 6:
        ticks = powers[(powers >= lo) & (powers <= hi)]
    if len(ticks) < 3 or len(ticks) > 6:
        locator = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
        ticks = locator.tick_values(lo, hi)
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]

    if all(1e-3 <= t < 1e4 for t in ticks):
        labels = [f'{t:.4g}' for t in ticks]
    else:
        labels = []
        for t in ticks:
            exponent = int(np.floor(np.log10(t)))
            mantissa = float(f'{t / 10. ** exponent:.3g}')
            if mantissa == 1:
                labels.append(f'$10^{{{exponent}}}$')
            else:
                labels.append(f'${mantissa:g} \\times 10^{{{exponent}}}$')

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_minor_locator(NullLocator())


def bottom_legend(fig, handles, labels=None, title=None, ncol=None, frameon=False, margin=0.05):
    """Add a single legend along the bottom of the figure, then lay out the panels above it

    `margin` is the fraction of the figure height reserved for the legend.
    """
    kwargs = {'handles': handles} if labels is None else {'handles': handles, 'labels': labels}
    fig.legend(**kwargs, title=title, loc='lower center', ncol=ncol or len(handles), frameon=frameon)
    fig.tight_layout(rect=(0, margin, 1, 1))



def use_paper_style():
    """Set matplotlib text and line sizes for figures drawn at the size they appear in the paper

    Use with figsize widths of TEXT_WIDTH, so that these are the sizes printed on the page.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6.5,
        'legend.title_fontsize': 6.5,
        'lines.linewidth': 1,
        'lines.markersize': 3,
        'axes.linewidth': 0.6,
    })
