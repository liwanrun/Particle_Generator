import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from rich import print as rprint
from rich import pretty
pretty.install()

def show_cmap(cmap, norm=None, extend=None):
    '''展示一个colormap.'''
    rprint("Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())
    if norm is None:
        norm = mcolors.Normalize(vmin=0, vmax=cmap.N)
    im = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    fig.colorbar(im, cax=ax, orientation='horizontal', extend=extend)
    plt.show()

if __name__ == '__main__':
    show_cmap(plt.get_cmap('jet', 8))