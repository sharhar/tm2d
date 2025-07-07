import numpy as np
import os
from datetime import datetime, timezone

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

from cmcrameri import cm

time_format = '%b%d_%H_%M_%S'
date_formatter = DateFormatter('%H:%M') # for plot axis



#%% GENERAL TOOLS

def today_str():
    return datetime.today().strftime('%y_%m_%d')


def make_cmap(rgb_path):
    rgb_values = []
    with open(rgb_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.startswith('ncolors'):
                continue
            try:
                r, g, b = map(float, line.strip().split())
                rgb_values.append((r, g, b))
            except ValueError:
                print('Skipping invalid line: {}'.format(line.strip()))
    cmap = mcolors.LinearSegmentedColormap.from_list(os.path.splitext(os.path.basename(rgb_path))[0], rgb_values)
    return cmap
# cmap_cividis = make_cmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cividis.rgb'))


def make_default_fig(figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax



#%% 2D PLOTTING TOOLS

def subplots(nrows=1, ncols=1, panel_length=4, padding=1):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*(panel_length + padding), nrows*panel_length))
    return fig, ax


def imshow_with_cbar(
    data, fig=None, ax=None, figsize=(5, 5), cmap=cm.grayC, cmap_type=None, title_str=None, ticks='off', labels=None,
    vmin=None, vmax=None, vcenter=None, deltav=None,
    ):

    if fig == None:
        fig, ax = make_default_fig(figsize=figsize)

    if vmin == None:
        vmin = data.min()
    if vmax == None:
        vmax = data.max()
    
    if vcenter is not None:
        if deltav == None:
            deltav = np.max([np.abs(vmax - vcenter), np.abs(vmin - vcenter)])
        vmin = vcenter - deltav
        vmax = vcenter + deltav

    if cmap_type == 'diverging':
        cmap = cm.vik
    elif cmap_type == 'cyclic':
        cmap = cm.vikO
        vmin = -np.pi
        vmax = np.pi
    elif cmap_type == 'sequential':
        cmap = cm.grayC
    elif cmap_type == 'batlow':
        cmap = cm.batlow
    
    if ticks != 'on':
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('scaled')

    if title_str is not None:
        ax.set_title(title_str)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    fig.colorbar(cax, ax=ax)
    fig.tight_layout()

    return (fig, ax)



#%% 1D PLOTTING TOOLS


def plot_time_trace(
    times, data, fig=None, ax=None, figsize=(10, 5), title_str=None, labels=None, xlim=None, ylim=None,
    alpha=0.5, linestyle='', marker='o', markerfacecolor='gray', markeredgecolor='k', grid='on'
    ):

    if fig == None:
        fig, ax = make_default_fig(figsize=figsize)

    if marker == 'o':
        ax.plot(times, data, c='k', linestyle=linestyle, marker='o', markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor, alpha=alpha)
    elif marker == 'x':
        ax.plot(times, data, c='k', linestyle=linestyle, marker='x', alpha=alpha)
    
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=90)

    if labels == None:
        ax.set_xlabel('time')
    else:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title_str is not None:
        ax.set_title(title_str)
    if grid == 'on':
        ax.grid()
    
    return (fig, ax)


def get_axes_with_histogram(fig=None, figsize=(10, 4)):
    if fig == None:
        fig = plt.figure(figsize=figsize)
    scatter_ax = fig.add_axes([0.1, 0.1, 0.6, 0.8]) # left, bottom, width, height
    hist_ax = fig.add_axes([0.71, 0.1, 0.2, 0.8]) # left, bottom, width, height
    return (fig, scatter_ax, hist_ax)
    
