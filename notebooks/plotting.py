from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy import special
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import igraph

from utils import degree_per_protein


COLOR_LIT = (60 / 255, 134 / 255, 184 / 255)
COLOR_LIT_BS = (217 / 255, 95 / 255, 2 / 255)
COLOR_LIT_NB = (149 / 255, 193 / 255, 30 / 255)
COLOR_SYS_NB = (228 / 255, 26 / 255, 28 / 255)
COLOR_Y2H = (155 / 255, 97 / 255, 153 / 255)
COLOR_RRS = "grey"
COLOR_PRS = (201 / 255, 104 / 255, 40 / 255)
COLOR_I3D = "orange"
COLOR_PREDICT = "pink"
COLOR_GI_PSN = (135 / 255, 200 / 255, 185 / 255)
COLOR_CS_PSN = (239 / 255, 177 / 255, 90 / 255)
COLOR_GE_PSN = (153 / 255, 192 / 255, 91 / 255)
COLOR_CYC2008 = "yellow"
COLOR_ZONE_A = (160 / 255, 54 / 255, 48 / 255)
COLOR_ZONE_B = (52 / 255, 106 / 255, 62 / 255)
COLOR_ZONE_C = (93 / 255, 167 / 255, 83 / 255)
COLOR_ZONE_D = (193 / 255, 212 / 255, 124 / 255)
COLOR_ALPHAFOLD_ROSETTAFOLD = "brown"


def psn_overlap_split_bar_and_pie_fig(
    ppi_nws,
    psn_nws,
    cat_column,
    cats,
    ppi_nw_labels,
    ppi_nw_colors,
    fig_name,
    ymax=1.0,
    xlabel_rotation=0,
    errorbar_capsize=0.75,
    cat_labels=None,
):
    if len(ppi_nw_labels) != len(ppi_nws):
        raise UserWarning("PPI NW labels argument not same size as number of PPI NWs")
    if len(ppi_nw_colors) != len(ppi_nws):
        raise UserWarning("PPI NW colors argument not same size as number of PPI NWs")
    if cat_labels is None:
        cat_labels = cats
    fig, axes = plt.subplots(len(psn_nws) + 1, len(ppi_nws))
    fig.set_size_inches(0.3 * len(ppi_nws) * (len(cats) + 1), 1.5 * (len(psn_nws) + 1))
    if len(cats) == 2:
        pie_colors = ["grey", "lightgrey"]
    elif len(cats) == 3:
        pie_colors = [
            "grey",
            "darkgrey",
            "lightgrey",
        ]  # NOTE: darkgrey lighter than grey
    elif len(cats) == 4:
        pie_colors = ["grey", "darkgrey", "lightgrey", "white"]
    else:
        pie_colors = None
    for nw, ax in zip(ppi_nws, axes[0, :]):
        patches, texts = ax.pie(
            nw[cat_column].value_counts()[cats],
            labels=cat_labels,
            colors=pie_colors,
            labeldistance=0.5,
            radius=1.5,
        )
        texts[0].set_color("white")
    for nw, color, ax_row in zip(ppi_nws, ppi_nw_colors, axes[1:, :].T):
        for psn, ax in zip(psn_nws, ax_row):
            validation_plot(
                data=nw,
                selections=[nw[cat_column] == x for x in cats],
                result_column="in_" + psn,
                colors=[color] * len(cats),
                ax=ax,
                labels=cat_labels,
                draw_numbers=False,
                errorbar_capsize=errorbar_capsize,
            )
    for ax in axes[1:, :].flatten():
        for x in ["top", "bottom", "right"]:
            ax.spines[x].set_visible(False)
        ax.set_facecolor("white")
        ax.xaxis.set_tick_params(length=0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        ax.set_ylim(0, ymax)
    for ax, psn in zip(axes[1:, 0], psn_nws):
        ax.set_ylabel("Fraction in " + psn.replace("_", "-"))
    for ax in axes[1:, 1:].flatten():
        ax.set_ylabel("")
    for ax in axes[1:-1, :].flatten():
        ax.set_xticks([])
    plt.subplots_adjust(wspace=0.9, hspace=0.2)
    for ax, label, color in zip(axes[0, :], ppi_nw_labels, ppi_nw_colors):
        ax.set_title(label, fontsize=13, color=color, fontweight="regular", pad=15)
    for ax in axes[-1, :].flatten():
        ax.xaxis.set_tick_params(rotation=xlabel_rotation)
    savefig(fig_name)
    return axes


def samogram(
    ppis,
    gene_property,
    n_bins,
    ax=None,
    id_a=None,
    id_b=None,
    correct_diagonal=False,
    self_interactions=True,
    draw_up=True,
    draw_right=False,
    draw_scale=True,
    reverse=False,
    vmax=None,
    ylim=None,
    ylabel=None,
    yticks=None,
    log=False,
    logy=False,
    zticks=None,
    cmap="Purples",
    color="purple",
    size_ratio=0.1,
    pad=0.04,
    colorbar_width=0.25,
    label=None,
    label_color=None,
    color_bar_label="Number of interactions",
    draw_red_triangle_at_fraction=None,
):
    """2D histogram of PPIs with genes ordered by a given property
    Bar charts of the mean of the property per bin can optionally be drawn.
    Gene property values are shuffled before sorting, so that genes with equal
    values are in a random order.
    See the :ref:`tutorial </samogram.ipynb>` for more information.
    Note:
        The diagonal bins have approximately half the number of possible
        interactions as the other bins, since the interactions are undirected.
        Specifically they have :math:`n^2/2 + n/2` possible interactions, with
        the other bins having :math:`n^2`, where :math:`n` is the number of
        genes per bin. This effect is counteracted by the much higher
        likelihood for genes to self-interact than interact with a
        random other gene. The smaller the bin size, the more
        self-interactions will have a visible impact.
    Args:
        ppis (pandas.DataFrame): table of interactions, one row per pair
        gene_property (pandas.Series): per-gene quantity to order proteins by
        n_bins (int): number of bins, heatmap will be n_bins x n_bins
        ax (matplotlib.axes.Axes): Axes to draw plot onto
        id_a/id_b (str): name of column containing gene/protein identifiers
        correct_diagonal (bool): increase number of interactions in diagonal
                                 bins to account for there being less possible
                                 combinations. Correction is dependent on value
                                 of `self_interactions` argument.
        self_interactions (bool): whether self interactions are plotted.
        draw_up/draw_right (bool): whether to add bar charts above / to the right
        draw_scale (bool): whether to draw the color scale
        reverse (bool): if true flips direction of ranking of genes
        vmax (int): upper limit on heatmap
        ylim ((float, float)): bar chart axis limits
        ylabel (str): bar chart axis label
        yticks/zticks (list(float)): bar chart / color scale axis ticks
        log/logy (bool): log scale for heatmap / bar charts
        cmap (str): colormap for heatmap
        color (str): color for bar charts
        size_ratio (float): fraction of size of bar charts to heatmap
        pad (float): space between bar charts and heatmap
        colorbar_width (float): size of colorbar as fraction of axes width
        label (str): label to put next to heatmap
    Returns:
        (list(matplotlib.axes.Axes), numpy.ndarray): new axes created, number of PPIs in each bin
    See Also:
        :func:`samogram_double`: plot two samograms in a single square
    Examples:
        Plot samogram of HI-III with default options:
        .. plot::
            :context: close-figs
            >>> import ccsblib.ccsbplotlib as cplt
            >>> from ccsblib import huri
            >>> hi_iii = huri.load_nw_hi_iii(id_type='ensembl_gene_id')
            >>> n_pub = huri.load_number_publications_per_gene()
            >>> cplt.samogram(hi_iii,
            ...               n_pub,
            ...               n_bins=40,
            ...               id_a='ensembl_gene_id_a',
            ...               id_b='ensembl_gene_id_b')
    """
    if size_ratio <= 0.0 or size_ratio >= 1.0:
        raise ValueError("size_ratio must be between 0 and 1")
    if ax is None:
        ax = plt.gca()
    if id_a is None:
        id_a = gene_property.index.name + "_a"
    if id_b is None:
        id_b = gene_property.index.name + "_b"
    cmap = plt.get_cmap(cmap)
    cmap.set_under(color=(0, 0, 0, 0))  # fully transparent
    genes_ranked = (
        gene_property.sample(frac=1, random_state=34832370)  # randomize order first
        .sort_values()
        .to_frame()
    )
    genes_ranked["rank"] = genes_ranked.reset_index().index.values
    binning = pd.qcut(genes_ranked["rank"], n_bins, labels=False)
    if reverse:
        binning = (n_bins - 1) - binning
    binned_ppis = ppis.loc[
        ppis[id_a].isin(gene_property.index)
        & ppis[id_b].isin(gene_property.index)
        & ((ppis[id_a] != ppis[id_b]) | self_interactions),
        [id_a, id_b],
    ].apply(lambda x: x.map(binning), axis=0)
    binned_ppis_x = binned_ppis.min(axis=1)
    binned_ppis_y = binned_ppis.max(axis=1)
    avrgs = gene_property.groupby(binning).mean()  # NOTE: this used to be median
    xlim = (n_bins - 0.5, -0.5)  # flip direction of x axis
    if ylabel is None and gene_property.name is not None:
        ylabel = gene_property.name.replace("_", "\n").capitalize()
    panel_ratios = [1 / size_ratio - 1.0, 1]
    gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=ax.get_subplotspec(),
        hspace=pad,
        wspace=pad,
        width_ratios=panel_ratios,
        height_ratios=panel_ratios[::-1],
    )
    ax_main = plt.subplot(gs.new_subplotspec((1, 0), rowspan=1, colspan=1))
    ax_main.set_anchor("NE")
    new_axes = [
        ax_main,
    ]
    if log:
        vmin = 1.0
    else:
        vmin = 0.0001
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    counts, _binsx, _binsy = np.histogram2d(
        binned_ppis_x,
        binned_ppis_y,
        bins=[i - 0.5 for i in range(n_bins + 1)],
    )
    if correct_diagonal:
        n = genes_ranked.shape[0] / n_bins
        if self_interactions:
            f = n**2 / (n**2 / 2 + n / 2)
        else:
            f = n**2 / (n**2 / 2 - n / 2)
        counts[np.diag_indices(counts.shape[0])] = counts.diagonal() * f
    img = ax_main.imshow(
        counts.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm
    )
    ax_main.spines["bottom"].set_visible(False)
    ax_main.spines["left"].set_visible(False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_xlim(xlim)
    if label is not None:
        if label_color is None:
            label_color = color
        ax_main.text(n_bins / 2.0, n_bins / 3.0, label, color=label_color, ha="right")
    if draw_up:
        ax_up = plt.subplot(gs.new_subplotspec((0, 0), rowspan=1, colspan=1))
        pos_ax_main = ax_main.get_position(original=False)
        pos_ax_up = ax_up.get_position(original=False)
        ax_up.set_position(
            [
                pos_ax_main.x0,
                pos_ax_up.y0,
                pos_ax_main.width,
                pos_ax_main.height * size_ratio,
            ]
        )
        new_axes.append(ax_up)
        ax_up.bar(x=avrgs.index, height=avrgs.values, width=1.0, color=color)
        ax_up.set_xlim(xlim)
        ax_up.set_xticks([])
        ax_up.spines["top"].set_visible(False)
        ax_up.spines["right"].set_visible(False)
        if logy:
            ax_up.set_yscale("log")
        ax_up.set_ylabel(ylabel)
        ax_up.set_ylim(ylim)
        if yticks is not None:
            ax_up.set_yticks(yticks)
        ax_up.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    if draw_right:
        ax_rt = plt.subplot(gs.new_subplotspec((1, 1), rowspan=1, colspan=1))
        pos_ax_main = ax_main.get_position(original=False)
        pos_ax_rt = ax_rt.get_position(original=False)
        ax_rt.set_position(
            [
                pos_ax_rt.x0,
                pos_ax_main.y0,
                pos_ax_main.width * size_ratio,
                pos_ax_main.height,
            ]
        )
        new_axes.append(ax_rt)
        ax_rt.barh(y=avrgs.index, width=avrgs.values, height=1.0, color=color)
        ax_rt.set_ylim((-0.5, n_bins - 0.5))
        ax_rt.set_yticks([])
        ax_rt.spines["bottom"].set_visible(False)
        ax_rt.spines["right"].set_visible(False)
        if logy:
            ax_rt.set_xscale("log")
        ax_rt.set_xlabel(ylabel)
        ax_rt.xaxis.set_label_position("top")
        ax_rt.set_xlim(ylim)
        if yticks is not None:
            ax_rt.set_xticks(yticks)
        ax_rt.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax_rt.xaxis.tick_top()
    if draw_scale:
        cax = inset_axes(
            ax_main,
            width="{:.0%}".format(colorbar_width),
            height="10%",
            loc="lower left",
        )
        if zticks is None:
            if log:
                zticks = [1, 100, 1000, 10000]
            else:
                zticks = [0, round(counts.max() / 2.0, 0), counts.max()]
        cb = plt.colorbar(
            img,
            cax=cax,
            orientation="horizontal",
            label=color_bar_label,
            ticks=zticks,
            format="%d",
        )
        cb.set_label(color_bar_label)
        if len(zticks) > 0:
            if zticks[0] == 0:
                cb.set_ticks([vmin] + zticks[1:])
                cb.set_ticklabels(
                    [str(int(zt)) for zt in zticks[:-1]] + ["≥" + str(int(zticks[-1]))]
                )  # HACK

    if draw_red_triangle_at_fraction is not None:
        q = draw_red_triangle_at_fraction
        df = ppis.copy()
        df["value_a"] = df[id_a].map(gene_property)
        df["value_b"] = df[id_b].map(gene_property)
        df["value_min"] = df[["value_a", "value_b"]].min(axis=1)
        xmin, xmax = ax_main.get_xlim()
        ymin, ymax = ax_main.get_ylim()
        f = (gene_property >= df["value_min"].quantile(1 - q)).mean()
        lw = 1.2
        ax_main.axvline(
            f * (xmax - xmin) + xmin,
            ymin=(1 - f) - 1 / n_bins,  # in axis fraction
            linewidth=lw,
            color="red",
            clip_on=False,
            solid_capstyle="round",
        )
        ax_main.axhline(
            ymax,
            xmax=f,
            xmin=-1 / n_bins,
            linewidth=lw,
            color="red",
            clip_on=False,
            zorder=999999999,
            solid_capstyle="round",
        )
        ax_main.plot(
            [xmin + 1, f * (xmax - xmin) + xmin],
            [ymax, ymax - f * (ymax - ymin) - 1],
            linewidth=lw,
            color="red",
            clip_on=False,
            solid_capstyle="round",
        )

    return new_axes, counts


def samogram_with_line(df, prop, q=0.80, **kwargs):
    new_axs, _binned_ppis = samogram(df, prop, **kwargs)
    df["n_pub_a"] = df["orf_name_a"].map(prop)
    df["n_pub_b"] = df["orf_name_b"].map(prop)
    df["n_pub_min"] = df[["n_pub_a", "n_pub_b"]].min(axis=1)
    xmin, xmax = new_axs[0].get_xlim()
    ymin, ymax = new_axs[0].get_ylim()
    f = (prop >= df["n_pub_min"].quantile(1 - q)).mean()
    lw = 1.2
    new_axs[0].axvline(
        f * (xmax - xmin) + xmin,
        ymin=(1 - f) - 1 / n_bins,  # in axis fraction
        linewidth=lw,
        color="red",
        clip_on=False,
        solid_capstyle="round",
    )
    new_axs[0].axhline(
        ymax,
        xmax=f,
        xmin=-1 / n_bins,
        linewidth=lw,
        color="red",
        clip_on=False,
        zorder=999999999,
        solid_capstyle="round",
    )
    new_axs[0].plot(
        [xmin + 1, f * (xmax - xmin) + xmin],
        [ymax, ymax - f * (ymax - ymin) - 1],
        linewidth=lw,
        color="red",
        clip_on=False,
        solid_capstyle="round",
    )


def samogram_double(
    ppis,
    gene_property,
    n_bins,
    ax=None,
    id_a=None,
    id_b=None,
    correct_diagonal=False,
    self_interactions=True,
    draw_up=True,
    draw_right=False,
    draw_scale=True,
    reverse=False,
    vmax=(None, None),
    ylim=None,
    ylabel=None,
    yticks=None,
    log=False,
    logy=False,
    zticks=(None, None),
    cmaps=("Purples", "Blues"),
    color="grey",
    size_ratio=0.1,
    pad=0.04,
    colorbar_width=0.25,
    labels=("", ""),
):
    """2D histogram of PPIs with genes ordered by a given property

    Bar charts of the median of the property per bin can optionally be drawn.
    Gene property values are shuffled before sorting, so that genes with equal
    values are in a random order.

    See the :ref:`tutorial </samogram.ipynb>` for more information.

    Note:
        The diagonal bins have approximately half the number of possible
        interactions as the other bins, since the interactions are undirected.
        Specifically they have :math:`n^2/2 + n/2` possible interactions, with
        the other bins having :math:`n^2`, where :math:`n` is the number of
        genes per bin. This effect is counteracted by the much higher
        likelihood for genes to self-interact than interact with a
        random other gene. The smaller the bin size, the more
        self-interactions will have a visible impact.

    Args:
        ppis (tuple(pandas.DataFrame, pandas.DataFrame)): two tables of interactions, one row per pair
        gene_property (pandas.Series): per-gene quantity to order proteins by
        n_bins (int): number of bins, heatmap will be n_bins x n_bins
        ax (matplotlib.axes.Axes): Axes to draw plot onto
        id_a/id_b (str): name of column containing gene/protein identifiers
        correct_diagonal (bool): increase number of interactions in diagonal
                                 bins to account for there being less possible
                                 combinations. Correction is dependent on value
                                 of `self_interactions` argument.
        self_interactions (bool): whether self interactions are plotted.
        draw_up/draw_right (bool): whether to add bar charts above / to the right
        draw_scale (bool): whether to draw the color scale
        reverse (bool): if true flips direction of ranking of genes
        vmax (int): upper limits on heatmap
        ylim ((float, float)): bar chart axis limits
        ylabel (str): bar chart axis label
        yticks (list(float)): bar chart axis ticks
        zticks (tuple(list(float), list(float))): color scale axis ticks
        log/logy (bool): log scale for heatmap / bar charts
        cmaps (tuple(str, str)): colormap for heatmap
        color (str): color for bar charts
        size_ratio (float): fraction of size of bar charts to heatmap
        pad (float): space between bar charts and heatmap
        colorbar_width (float): size of colorbar as fraction of axes width
        labels (tuple(str, str)): label to put next to heatmap

    Returns:
        (list(matplotlib.axes.Axes)): new axes created

    See Also:
        :func:`samogram`: individual samogram

    Examples:
        Plot samogram of HI-III with default options:

        .. plot::
            :context: close-figs

            >>> import ccsblib.ccsbplotlib as cplt
            >>> from ccsblib import huri
            >>> hi_iii = huri.load_nw_hi_iii(id_type='ensembl_gene_id')
            >>> lit_bm = huri.load_nw_lit_bm_17(id_type='ensembl_gene_id')
            >>> n_pub = huri.load_number_publications_per_gene()
            >>> cplt.samogram_double((hi_iii, lit_bm),
            ...                      n_pub,
            ...                      n_bins=40,
            ...                      id_a='ensembl_gene_id_a',
            ...                      id_b='ensembl_gene_id_b',
            ...                      color='grey',
            ...                      draw_right=True,
            ...                      cmaps=('Purples', 'Blues'),
            ...                      zticks=([0, 100, 200],
            ...                              [0, 100, 200]),
            ...                      vmax=(200, 200),
            ...                      ylabel='Number of publications',
            ...                      labels=('HI-III', 'Lit-BM'))
            >>> import matplotlib.pyplot as plt; fig = plt.gcf(); fig.set_size_inches(6, 6)

    """
    if size_ratio <= 0.0 or size_ratio >= 1.0:
        raise ValueError("size_ratio must be between 0 and 1")
    if ax is None:
        ax = plt.gca()
    if id_a is None:
        id_a = gene_property.index.name + "_a"
    if id_b is None:
        id_b = gene_property.index.name + "_b"
    if isinstance(vmax, (float, int)):
        vmax = (vmax, vmax)
    if isinstance(cmaps, str):
        cmaps = (cmaps, cmaps)
    if isinstance(zticks[0], (float, int)):
        zticks = (zticks, zticks)
    cmap_a = plt.get_cmap(cmaps[0])
    cmap_b = plt.get_cmap(cmaps[1])
    cmap_a.set_under(color=(0, 0, 0, 0))  # fully transparent
    cmap_b.set_under(color=(0, 0, 0, 0))
    genes_ranked = (
        gene_property.sample(frac=1).sort_values().to_frame()  # randomize order first
    )
    genes_ranked["rank"] = genes_ranked.reset_index().index.values
    binning = pd.qcut(genes_ranked["rank"], n_bins, labels=False)
    if reverse:
        binning = (n_bins - 1) - binning
    binned_ppis_a = (
        ppis[0]
        .loc[
            ppis[0][id_a].isin(gene_property.index)
            & ppis[0][id_b].isin(gene_property.index)
            & ((ppis[0][id_a] != ppis[0][id_b]) | self_interactions),
            [id_a, id_b],
        ]
        .apply(lambda x: x.map(binning), axis=0)
    )
    binned_ppis_a_x = binned_ppis_a.min(axis=1)
    binned_ppis_a_y = binned_ppis_a.max(axis=1)
    binned_ppis_b = (
        ppis[1]
        .loc[
            ppis[1][id_a].isin(gene_property.index)
            & ppis[1][id_b].isin(gene_property.index)
            & ((ppis[1][id_a] != ppis[1][id_b]) | self_interactions),
            [id_a, id_b],
        ]
        .apply(lambda x: x.map(binning), axis=0)
    )
    binned_ppis_b_x = binned_ppis_b.min(axis=1)
    binned_ppis_b_y = binned_ppis_b.max(axis=1)
    avrgs = gene_property.groupby(binning).median()
    xlim = (n_bins - 0.5, -0.5)  # flip direction of x axis
    if ylabel is None and gene_property.name is not None:
        ylabel = gene_property.name.replace("_", "\n").capitalize()
    panel_ratios = [1 / size_ratio - 1.0, 1]
    # cut out sqaure sub-area of given Axes
    # get size in inches and convert to figure fraction
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_scale = min(bbox.width, bbox.height) / bbox.width
    height_scale = min(bbox.width, bbox.height) / bbox.height
    x0, y0, x1, y1 = ax.figbox.get_points().flatten()
    x0 = x0 + (x1 - x0) * (1 - width_scale) / 2
    x1 = x1 - (x1 - x0) * (1 - width_scale) / 2
    y0 = y0 + (y1 - y0) * (1 - height_scale) / 2
    y1 = y1 - (y1 - y0) * (1 - height_scale) / 2
    gs = gridspec.GridSpec(
        2,
        2,
        left=x0,
        right=x1,
        bottom=y0,
        top=y1,
        hspace=pad,
        wspace=pad,
        width_ratios=panel_ratios,
        height_ratios=panel_ratios[::-1],
    )
    ax_main = plt.subplot(gs[1, 0])
    new_axes = [ax_main]
    if log:
        vmin = 1.0
    else:
        vmin = 0.0001
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    counts_a, _binsx, _binsy = np.histogram2d(
        binned_ppis_a_x,
        binned_ppis_a_y,
        bins=[i - 0.5 for i in range(n_bins + 1)],
    )
    # Note the reversal of x and y below
    counts_b, _binsx, _binsy = np.histogram2d(
        binned_ppis_b_y,
        binned_ppis_b_x,
        bins=[i - 0.5 for i in range(n_bins + 1)],
    )
    if correct_diagonal:
        n = genes_ranked.shape[0] / n_bins
        if self_interactions:
            f = n**2 / (n**2 / 2 + n / 2)
        else:
            f = n**2 / (n**2 / 2 - n / 2)
        counts_a[np.diag_indices(counts_a.shape[0])] = counts_a.diagonal() * f
        counts_b[np.diag_indices(counts_b.shape[0])] = counts_b.diagonal() * f
    img_a = ax_main.imshow(
        counts_a.T, origin="lower", cmap=cmap_a, vmin=vmin, vmax=vmax[0], norm=norm
    )
    img_b = ax_main.imshow(
        counts_b.T, origin="lower", cmap=cmap_b, vmin=vmin, vmax=vmax[1], norm=norm
    )
    xhigh, xlow = ax_main.get_xlim()
    ylow, yhigh = ax_main.get_ylim()
    crop_triangle_a = mpl.patches.Polygon(
        [[xlow, yhigh], [xhigh, ylow], [xhigh, yhigh]], transform=ax_main.transData
    )
    img_a.set_clip_path(crop_triangle_a)
    crop_triangle_b = mpl.patches.Polygon(
        [[xlow, yhigh], [xhigh, ylow], [xlow, ylow]], transform=ax_main.transData
    )
    img_b.set_clip_path(crop_triangle_b)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_xlim(xlim)
    # diagonal dividing line
    ax_main.plot(
        ax_main.get_xlim()[::-1],
        ax_main.get_ylim(),
        color=ax_main.spines["top"].get_edgecolor(),
        linewidth=ax_main.spines["top"].get_linewidth(),
        linestyle=ax_main.spines["top"].get_linestyle(),
    )
    if draw_up:
        ax_up = plt.subplot(gs[0, 0], sharex=ax_main)
        new_axes.append(ax_up)
        ax_up.bar(x=avrgs.index, height=avrgs.values, width=1.0, color=color)
        ax_up.set_xlim(xlim)
        ax_up.set_xticks([])
        ax_up.spines["top"].set_visible(False)
        ax_up.spines["right"].set_visible(False)
        if logy:
            ax_up.set_yscale("log")
        ax_up.set_ylabel(ylabel)
        ax_up.set_ylim(ylim)
        if yticks is not None:
            ax_up.set_yticks(yticks)
        ax_up.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    if draw_right:
        ax_rt = plt.subplot(gs[1, 1], sharey=ax_main)
        new_axes.append(ax_rt)
        ax_rt.barh(y=avrgs.index, width=avrgs.values, height=1.0, color=color)
        ax_rt.set_ylim((-0.5, n_bins - 0.5))
        ax_rt.set_yticks([])
        ax_rt.spines["bottom"].set_visible(False)
        ax_rt.spines["right"].set_visible(False)
        if logy:
            ax_rt.set_xscale("log")
        ax_rt.set_xlabel(ylabel)
        ax_rt.xaxis.set_label_position("top")
        ax_rt.set_xlim(ylim)
        if yticks is not None:
            ax_rt.set_xticks(yticks)
        ax_rt.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax_rt.xaxis.tick_top()
    if draw_scale:
        zticks_a, zticks_b = zticks
        same_scale = vmax[0] == vmax[1]
        # I don't understand why I have to put 0.25 instead of 0.1....
        cax_a = inset_axes(
            ax_main,
            width="{:.0%}".format(colorbar_width),
            height="10%",
            loc="lower center",
            bbox_to_anchor=[0.0, -0.35, 1.0, 1.0],
            bbox_transform=ax_main.transAxes,
        )
        cax_b = inset_axes(
            ax_main,
            width="{:.0%}".format(colorbar_width),
            height="10%",
            loc="lower center",
            bbox_to_anchor=[0.0, -0.25, 1.0, 1.0],
            bbox_transform=ax_main.transAxes,
        )
        if zticks_a is None:
            if log:
                zticks_a = [1, 100, 1000, 10000]
            else:
                zticks_a = [0, round(counts_a.max() / 2.0, 0), counts_a.max()]
        if zticks_b is None:
            if log:
                zticks_b = [1, 100, 1000, 10000]
            else:
                zticks_b = [0, round(counts_b.max() / 2.0, 0), counts_b.max()]
        if same_scale:
            zticks_b = []
        cb_a = plt.colorbar(
            img_a,
            cax=cax_a,
            orientation="horizontal",
            label="Number of interactions",
            ticks=zticks_a,
            format="%d",
        )
        cb_b = plt.colorbar(
            img_b, cax=cax_b, orientation="horizontal", label="", ticks=zticks_b
        )
        cax_b.xaxis.set_ticks_position("top")
        # cax_a.text(cb_a.get_clim()[1] * 1.05, 1, labels[0])
        # cax_b.text(cb_b.get_clim()[1] * 1.05, 1, labels[1])
        if len(zticks_a) > 0:
            if zticks_a[0] == 0:
                cb_a.set_ticks([vmin] + zticks_a[1:])
            tick_labels = [str(int(zt)) for zt in zticks_a]
            if vmax[0] is not None:
                if vmax[0] < counts_a.max() or (
                    (vmax[0] < counts_b.max()) and same_scale
                ):
                    tick_labels = tick_labels[:-1] + ["≥" + tick_labels[-1]]
            cb_a.set_ticklabels(tick_labels)
        if len(zticks_b) > 0 and not same_scale:
            if zticks_b[0] == 0:
                cb_b.set_ticks([vmin] + zticks_b[1:])
            tick_labels = [str(int(zt)) for zt in zticks_b]
            if vmax[1] is not None:
                if vmax[1] < counts_b.max() and not same_scale:
                    tick_labels = tick_labels[:-1] + ["≥" + tick_labels[-1]]
            cb_b.set_ticklabels(tick_labels)
    return new_axes


def checkerboard(
    data,
    protein_a_column=None,
    protein_b_column=None,
    detection_columns=None,
    sort=True,
    alternative_sort=False,
    assay_labels=None,
    colors=None,
    draw_gene_names=True,
    draw_box=False,
    draw_grid=False,
    grid_color="white",
    ax=None,
):
    """Plot yes/no detection for benchmark PPI set with different assays

    See Braun et al, Nature Methods, 2010 for examples.

    Args:
        data (pandas.DataFrame): PPI test results. No missing values.
        protein_a/b_column (str): columns with protein names
        detection_columns (list(str)): name of columns containing boolean results
        sort (bool): whether to sort pairs by number of assays detected and assay order
        assay_labels (list(str)): names of assays to print
        positive_color (str/RGB/RGBA or list(colors)): single color or list of colors for each different assay
        negative_color (str/RGB/RGBA): color to indicate undetected pairs
        ax (matplotlib.axes.Axes): Axes to draw plot onto

    Examples:
        Make a checkerboard of some dummy data:

        .. plot::
            :context: close-figs

            >>> import pandas as pd
            >>> import ccsblib.ccsbplotlib as cplt
            >>> prs_results = pd.DataFrame(columns=['gene_a', 'gene_b', 'Y2H', 'MAPPIT', 'GPCA'],
            ...                            data=[['ABC1', 'ABC2', False, False, False],
            ...                                  ['EFG1', 'EFG2', False, False, False],
            ...                                  ['HIJ1', 'HIJ2', True, False, False],
            ...                                  ['KLM1', 'KLM2', False, False, False],
            ...                                  ['NOP1', 'NOP2', True, True, True],
            ...                                  ['QRS1', 'QRS2', True, False, True],
            ...                                  ['TUV1', 'TUV2', False, False, True],
            ...                                  ['XYZ1', 'XYZ2', False, False, False]])
            >>> cplt.checkerboard(data=prs_results,
            ...                   protein_a_column='gene_a',
            ...                   protein_b_column='gene_b',
            ...                   detection_columns=['Y2H',
            ...                                      'MAPPIT',
            ...                                      'GPCA'])

    """
    df = data.copy()
    if ax is None:
        ax = plt.gca()
    if assay_labels is None:
        assay_labels = detection_columns
    if protein_a_column is None:
        protein_a_column = df.columns[0]
    if protein_b_column is None:
        protein_a_column = df.columns[1]
    if detection_columns is None:
        detection_columns = list(df.columns[df.dtypes == bool])
    elif isinstance(detection_columns, str):
        detection_columns = [detection_columns]
    df["total_positives"] = (
        (df[detection_columns] == "Positive").sum(axis=1)
        + (df[detection_columns] == "Negative").sum(axis=1) / 100.0
        + (df[detection_columns] == "Autoactivator").sum(axis=1) / 1000.0
        + (df[detection_columns] == "Failed sequence confirmation").sum(axis=1)
        / 10000.0
        + (df[detection_columns] == "Test failed").sum(axis=1) / 100000.0
    )
    sort_values = {
        "Positive": 1,
        "Negative": 0,
        "Test failed": 0,
        "Failed sequence confirmation": 0,
        "Autoactivator": 0,
    }
    if sort:
        df = df.sort_values(by=["total_positives"] + detection_columns, ascending=False)
    if alternative_sort:
        df = df.sort_values(
            by=detection_columns + ["total_positives"],
            key=lambda x: x.map(lambda y: sort_values[y]) if x.dtype == "O" else x,
            ascending=False,
        )

    possible_values = list(set(df[detection_columns].values.flatten()))
    ax.imshow(
        df[detection_columns].applymap(lambda x: possible_values.index(x)).values.T,
        aspect="auto",
        cmap=mpl.colors.ListedColormap([colors[v] for v in possible_values]),
    )

    if not draw_box:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    ax.set_yticks(range(len(detection_columns)))
    ax.set_yticklabels(assay_labels, fontsize=6)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xticks([])
    if draw_grid:
        ax.set_xticks(np.linspace(0.5, df.shape[0] - 1.5, df.shape[0] - 1), minor=True)
        ax.set_yticks(
            np.linspace(0.5, len(detection_columns) - 1.5, len(detection_columns) - 1),
            minor=True,
        )
        ax.xaxis.set_tick_params(length=0, which="minor")
        ax.yaxis.set_tick_params(length=0, which="minor")
        ax.grid(color=grid_color, axis="both", which="minor", zorder=5, linewidth=0.5)
    if draw_gene_names:
        len_longest_name = df[protein_a_column].str.len().max()
        for i, (name_a, name_b) in enumerate(
            zip(df[protein_a_column].values, df[protein_b_column].values)
        ):
            ax.text(
                i,
                -0.6,
                name_a + " " * (len_longest_name - len(name_a) + 2) + name_b,
                rotation=90,
                fontfamily="monospace",
                fontsize=5,
                va="bottom",
                ha="center",
            )
    # ax.set_ylim(-0.5, len(detection_columns) - 0.5)  # Fixing a problem with new matplotlib
    return [
        name_a + " " + name_b
        for name_a, name_b in zip(
            df[protein_a_column].values, df[protein_b_column].values
        )
    ]


def validation_plot(
    positives=None,
    n_tested=None,
    data=None,
    selections=None,
    result_column="result",
    labels=None,
    colors=None,
    ax=None,
    bayes_errors=True,
    y_max=1.0,
    draw_numbers=True,
    xlabel_rotation=0,
    errorbar_capsize=0.9,
    errorbar_thickness=None,
    bar_spacing=0.2,
):
    """Compare the validation rate of different cateogies.

    Missing values are not used in the denominator.

    See the :ref:`tutorial </validation_plots.ipynb>` for more information.

    Args:
        positives (list(int)): number tested positive in each category
        n_tested (list(int)): number successfully tested in each category
        data (pandas.DataFrame): results of validation experiment
        selections (list(pandas.Series)): boolean rows index for each category
        ax (matplotlib.axes.Axes): Axes to draw plot onto
        bayes_errors (bool): do Bayesian error bars, if false use standard error
                            on proportion
        result_column (str): column containing 0/1/nan for result of test
        labels (list(str)): name of each category
        colors (list(str)): color for each bar
        y_max (float): y axis upper limit
        draw_numbers (bool): flag to print the numbers on top of the bars
        xlabel_rotation (float): rotating the x axis labels
        errorbar_capsize (float): as fraction of the width of the bars
        errorbar_thickness (float): width of error bar lines
        bar_spacing (float): must be between 0 and 1

    Examples:
        There are two ways to call the function. Either give it the raw data:



        .. plot::
            :context: close-figs

            >>> import ccsblib.ccsbplotlib as cplt
            >>> cplt.validation_plot(positives=[20, 1, 19],
            ...                      n_tested=[100, 100, 100],
            ...                      labels=['PRS', 'RRS', 'Y2H'],
            ...                      y_max=0.3)

        Or pass it the validation results as a DataFrame and a list of the rows
        for each category:

        .. plot::
            :context: close-figs

            >>> from ccsblib import huri
            >>> data = huri.load_GPCA_and_MAPPIT_data()
            >>> exp = (data['assay'] == 'MAPPIT') & (data['standard_batch'] == 'Hvs01')
            >>> sources = ['lit_bm_2013_rand250', 'Hs01', 'RRS']
            >>> categories = [exp & (data['source'] == cat) for cat in sources]
            >>> cplt.validation_plot(data=data,
            ...                      selections=categories,
            ...                      labels=sources,
            ...                      y_max=0.25)

    """
    signature_a = positives is not None and n_tested is not None
    signature_b = data is not None and selections is not None
    if signature_a == signature_b:
        msg = """Must supply only one of both positives and n_tested or
                 both data and selections."""
        raise ValueError(msg)
    if signature_b:
        if (
            not data.loc[data[result_column].notnull(), result_column]
            .isin({0, 1})
            .all()
        ):
            raise ValueError("Only expect 0/1/missing in result column")
        positives = [
            (data.loc[rows, :][result_column] == 1).sum() for rows in selections
        ]
        n_tested = [
            data.loc[rows, :][result_column].notnull().sum() for rows in selections
        ]
    _validation_plot(
        positives=positives,
        n_tested=n_tested,
        colors=colors,
        labels=labels,
        ax=ax,
        bayes_errors=bayes_errors,
        y_max=y_max,
        draw_numbers=draw_numbers,
        xlabel_rotation=xlabel_rotation,
        errorbar_capsize=errorbar_capsize,
        errorbar_thickness=errorbar_thickness,
        bar_spacing=bar_spacing,
    )


def _validation_plot(
    positives,
    n_tested,
    colors=None,
    labels=None,
    ax=None,
    bayes_errors=True,
    y_max=1.0,
    draw_numbers=True,
    xlabel_rotation=0.0,
    errorbar_capsize=5,
    errorbar_thickness=None,
    bar_spacing=0.2,
):
    if len(positives) != len(n_tested):
        raise ValueError("Lengths of positives and n_tested must be equal")
    if any([p > n for p, n in zip(positives, n_tested)]):
        raise ValueError("Number of positives must be <= number tested")
    if bar_spacing > 1.0 or bar_spacing < 0.0:
        msg = "bar_spacing={}\nbar_spacing must be between 0 and 1"
        msg = msg.format(bar_spacing)
        raise ValueError(msg)
    bar_width = 1.0 - bar_spacing
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = [""] * len(positives)
    if colors is None:
        colors = [None] * len(positives)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1), minor=False)
    ax.set_yticks(np.arange(0.05, 1.0, 0.1), minor=True)
    ax.set_facecolor("0.96")
    ax.set_axisbelow(True)
    ax.grid(color="white", axis="y", which="both", zorder=5)
    pos = np.array(positives)
    tested = np.array(n_tested)
    neg = tested - pos
    fracs = pos / tested
    if bayes_errors:
        intv = stats.beta.interval(0.6827, pos + 1, neg + 1)
        errs = [fracs - intv[0], intv[1] - fracs]
        errs[0][pos == 0] = 0.0
        errs[1][neg == 0] = 0.0
    else:
        stdErrProp = np.sqrt((fracs * (1.0 - fracs)) / (pos + neg))
        errs = [stdErrProp, stdErrProp]
    for i in range(len(positives)):
        ax.bar(i, fracs[i], color=colors[i], label=labels[i], width=bar_width)
        if draw_numbers:
            c = "white"
            h = 0.02  # default height to draw numbers
            if fracs[i] < h:
                c = "black"
            if (errs[1][i] + fracs[i]) > h and (fracs[i] - errs[0][i]) < (h + 0.04):
                c = "black"
                h = fracs[i] + errs[1][i] + 0.02
            ax.text(i, h, "{}/{}".format(pos[i], pos[i] + neg[i]), color=c, ha="center")

    ax.figure.canvas.draw()  # Need to draw figure to get correct values below
    bar_width_pixels = (
        ax.transData.transform((bar_width, 0)) - ax.transData.transform((0, 0))
    )[0]
    # 72 comes from definition of a point as 1 / 72 inches
    bar_width_points = (72.275 / ax.figure.dpi) * bar_width_pixels

    ax.errorbar(
        range(fracs.shape[0]),
        fracs,
        yerr=errs,
        color="black",
        fmt="none",
        # factor of 0.5 because the size is measured from the center to the end
        # i.e. full size is twice that
        capsize=bar_width_points * errorbar_capsize * 0.5,
        elinewidth=errorbar_thickness,
        capthick=errorbar_thickness,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=xlabel_rotation)
    ax.set_ylim((0.0, y_max))
    ax.set_ylabel("Fraction positive")
    if isinstance(ax.containers[-1], mpl.container.ErrorbarContainer):
        bottom_cap, top_cap = ax.containers[-1].lines[1]
        bottom_cap.set_data(
            (
                [x for x, y in zip(*bottom_cap.get_data()) if y > 0],
                [y for x, y in zip(*bottom_cap.get_data()) if y > 0],
            )
        )
        top_cap.set_data(
            (
                [x for x, y in zip(*top_cap.get_data()) if y < 1],
                [y for x, y in zip(*top_cap.get_data()) if y < 1],
            )
        )
    else:
        warnings.warn("problem with code to remove lower error bar caps for 0 counts")


def validation_titration_plot(
    data,
    selections,
    threshold=None,
    xmin=None,
    xmax=None,
    ymax=None,
    score_column="score",
    labels=None,
    colors=None,
    line_styles=None,
    ax=None,
    threshold_label=None,
    threshold_color="grey",
    plot_kwargs=None,
):
    """ """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = [""] * len(selections)
    if colors is None:
        colors = [None] * len(selections)
    if line_styles is None:
        line_styles = ["-"] * len(selections)
    if xmin is None:
        xmin = data[score_column].min()
    if xmax is None:
        xmax = data[score_column].max()
    n_points = 1000  # TODO: this is a bad way to do it, would be better to just get every point where there is a pair
    points = np.linspace(xmin, xmax, n_points)
    for selection, label, color, line_style in zip(
        selections, labels, colors, line_styles
    ):
        n = data.loc[selection, score_column].notnull().sum()
        pos = np.array([(data.loc[selection, score_column] > x).sum() for x in points])
        neg = n - pos
        fracs = pos / n
        ax.plot(
            points, fracs, color=color, label=label, linestyle=line_style, **plot_kwargs
        )
        intv = stats.beta.interval(0.6827, pos + 1, neg + 1)
        errs = [fracs - intv[0], intv[1] - fracs]
        errs[0][pos == 0] = 0.0
        errs[1][neg == 0] = 0.0
        ax.fill_between(
            points,
            fracs - errs[0],
            fracs + errs[1],
            color=color,
            alpha=0.2,
            linewidth=0,
        )
    ax.set_ylim(0, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel("Fraction positive")
    ax.set_xlabel("Score threshold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    if threshold is not None:
        ax.axvline(
            x=threshold,
            ymin=0,
            ymax=1,
            linestyle="--",
            color=threshold_color,
            linewidth=1,
        )
        if threshold_label is not None:
            ax.text(
                x=threshold + (xmax - xmin) * 0.02,
                y=ymax,
                s=threshold_label,
                color=threshold_color,
                verticalalignment="top",
                horizontalalignment="left",
                fontsize=8,
            )


def validation_by_degree_plot(
    data,
    x_name,
    n_bins=10,
    ax=None,
    y_name="result",
    regression=True,
    ymax=1.0,
    draw_points=False,
    color=None,
    line_color=None,
    ylabel="validation fraction",
    xlabel=None,
    logx=False,
):
    """Plot validation results binned in quantiles of degree.

    An optional logistic regression can be drawn. It uses the
    log-transformed degrees.

    Args:
        data (pandas.DataFrame): validation data, one row per pair
        x_name (str): name of degree columns
        n_bins (int): number of bins
        y_name (str): name of binary validation result column
        ax (matplotlib.axes.Axes): axes to draw plot onto
        regression (bool): option to draw regression line
        ymax (float): y-axis upper limit
        draw_points (bool): display data points as ticks on the top and bottom
                            of the axes
        color (str): color of points
        line_color (str): color of regression line
        ylabel (str): axis label
        xlabel (str): axis label
        logx (bool): log-scale for x-axis

    Examples:
        Validation rate vs. higher of the two degrees:

        .. plot::
            :context: close-figs

            >>> import ccsblib.ccsbplotlib as cplt
            >>> from ccsblib import huri
            >>> data = huri.load_GPCA_and_MAPPIT_data()
            >>> orfs = huri.load_orfs_hi_iii()
            >>> data['degree_a'] = data['orf_id_a'].map(orfs['degree'])
            >>> data['degree_b'] = data['orf_id_b'].map(orfs['degree'])
            >>> data['max_degree'] = data[['degree_a', 'degree_b']].max(axis=1)
            >>> selection = ((data['assay'] == 'MAPPIT') &
            ...              (data['standard_batch'] == 'Hvs06') &
            ...              (data['source'] == 'Hs09') &
            ...              data['max_degree'].notnull())
            >>> cplt.validation_by_degree_plot(data.loc[selection, :], 'max_degree', ymax=0.4)

    """
    import statsmodels.api as sm

    if ax is None:
        ax = plt.gca()
    val = data.copy()
    val = val.loc[val[y_name].notnull(), :]
    if val[x_name].isnull().any():
        warnings.warn("Missing values in " + x_name)
        val = val.loc[val[x_name].notnull(), :]
    binned, bins = pd.qcut(val[x_name], n_bins, retbins=True)
    pos = val.groupby(binned)[y_name].sum().values
    neg = val.groupby(binned)[y_name].size().values - pos
    fracs = pos / (pos + neg)
    intv = stats.beta.interval(0.6827, pos + 1, neg + 1)
    errs = [fracs - intv[0], intv[1] - fracs]
    errs[0][pos == 0] = 0.0
    errs[1][neg == 0] = 0.0
    bin_means = val.groupby(binned)[x_name].mean().values
    ax.errorbar(bin_means, fracs, yerr=errs, fmt="o", color=color)

    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    if regression:
        glm = sm.GLM(
            val[y_name],
            sm.add_constant(np.log2(val[x_name])),
            family=sm.families.Binomial(),
        )
        res = glm.fit()
        beta = res.params.values
        cov = res.cov_params().values
        xs = np.linspace(val[x_name].min(), val[x_name].max(), 1000)
        X = sm.add_constant(np.log2(xs))
        fit_line = [logistic(np.matmul(beta, x)) for x in X]
        upper = [
            logistic(
                np.matmul(beta, x) + 1.96 * np.sqrt(np.matmul(np.matmul(x, cov), x))
            )
            for x in X
        ]
        lower = [
            logistic(
                np.matmul(beta, x) - 1.96 * np.sqrt(np.matmul(np.matmul(x, cov), x))
            )
            for x in X
        ]
        ax.plot(xs, fit_line, "--", color=line_color)
        ax.fill_between(xs, upper, lower, color=line_color, alpha=0.2)
    ax.set_ylabel(ylabel)
    if xlabel is None:
        xlabel = x_name.replace("_", " ")
    ax.set_xlabel(xlabel)
    ax.set_ylim((0.0, ymax))
    if draw_points:
        for result, degree in val[[y_name, x_name]].values:
            ax.axvline(degree, result, abs(result - 0.05), linewidth=1)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))


def degree_distribution_plot(nw,
                             id_a, id_b,
                             ax=None,
                             ymin=None, xmax=None,
                             do_fit=True,
                             print_fit_summary=True,
                             grey_dot_size=None,
                             **kwargs):
    # TODO:
    # error on gamma value
    # some kind of minimum quality criteria for fit?
    if ax is None:
        ax = plt.gca()
    d = degree_per_protein(nw, id_a, id_b)
    if do_fit:
        fit = igraph.power_law_fit(d.values, method='discrete')
        if print_fit_summary:
            print(fit.summary())
    pk_individual = d.value_counts() / d.shape[0]
    ax.scatter(pk_individual.index, 
               pk_individual.values,
               color='lightgrey',
               clip_on=False,
               s=grey_dot_size)    
    if do_fit:
        # equation 4.42
        scaling_factor = (1 / special.zeta(fit.alpha, fit.xmin)) * ((d >= fit.xmin).sum() / d.shape)     
        ax.plot([fit.xmin, d.max()],
                [(fit.xmin ** -fit.alpha) * scaling_factor,
                (d.max() ** -fit.alpha) * scaling_factor],
                linestyle='--',
                color='black')
    bins = []
    i = 0
    while 2**i - 0.5 < d.max():
        bins.append(2**i - 0.5)
        i += 1
    #pk = ((d.groupby(pd.cut(d, bins)).size().values /
    #       np.array([bins[j + 1] - bins[j] for j in range(len(bins) - 1)])) /
    #      d.shape[0])
    pk = np.histogram(d.values, bins, density=True)[0]
    dMeans = d.groupby(pd.cut(d, bins)).mean().values
    ax.plot(dMeans, 
            pk,
            'o',
            clip_on=False,
            **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')  
    ax.set_ylim((min([pk.min(), pk_individual.min()]) * 0.75 if ymin is None else ymin, 1.0))
    ax.set_xlim((0.8, xmax))
    ax.set_ylabel('Probability density')
    ax.set_xlabel('Degree')
    ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()])
    if do_fit:
        ax.text(0.15, 0.15,
            r'$\gamma = {:.1f}$'.format(fit.alpha),
            transform=ax.transAxes)


def savefig(file_name):
    plt.savefig(
        os.path.join("../figures/", file_name + ".pdf"),
        metadata={
            "Creator": None,  # make PDFs deterministic
            "Producer": None,  # so the hashes can be compared
            "CreationDate": None,
        },  # to see if anything changed
        dpi=500,
        bbox_inches="tight",
    )
