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
    # show_cmap(plt.get_cmap('jet', 8))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'   # 设置字体系列为 serif
    plt.rcParams['font.serif'] = ['Times']  # 设置 serif 字体为 Times
    plt.rcParams['font.size'] = 14
    plt.style.use('seaborn-v0_8')
    #print(plt.style.available)

    n = 0.5
    d_min = 0.4
    d_max = 1.0
    d = np.linspace(0.4, 1.0, 100)
    p1 = 100.0*np.power(d / d_max, 0.3)
    p2 = 100.0*np.power(d / d_max, n)
    print(d[33])
    p3 = 100.0*np.power(d / d_max, 0.7)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), layout='constrained')
    line_1 = axs.plot(d, p1, '-', lw=2.0, label='n = 0.3')
    line_2 = axs.plot(d, p2, lw=3.0, label='n = 0.5')
    line_3 = axs.plot(d, p3, '-', lw=2.0, label='n = 0.7')
    axs.set_title(r"Fuller's grading curve: $P(d) = 100(d/d_{max})^{n}$", fontsize=14)
    axs.set_xlabel(r'Gravel size $d (m)$', fontsize=14)
    axs.set_ylabel(r'Cumulative percentage $P(\%)$', fontsize=14)
    axs.set_xlim([0.0, 1.0])
    #axs.set_xticks(ticks=[0.4, 1.0], labels=[r'$d_{min}$', r'$d_{max}$'])
    axs.set_ylim([p3[0], 100.0])
    axs.legend(fontsize=12)
    axs.fill_between(d, p2, color='C1', alpha=0.2)
    #axs.hlines(y=[p2[0]], xmin=0.0, xmax=d_min, color='C1', ls='--')
    #axs.hlines(y=[p2[-1]], xmin=0.0, xmax=d_max, color='C1')
    axs.set_box_aspect(1.0)
    axs.stem([0.4, 0.6, 1.0], [p2[0], p2[33], p2[-1]], 'C1:')
    #axs.grid(True)
    plt.show()
    fig.savefig('grading_curve.svg', transparent=False)
