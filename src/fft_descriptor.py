# Visiualization of the relationship between Fourier descriptors and particle morphology
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif=['Times'])
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
#matplotlib.rcParams['font.size'] = 14
plt.style.use('seaborn-v0_8')

fs = 15
nfreq = 128
spectrums = np.zeros((2, 6, nfreq), dtype='complex')

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 6.2), sharex=True, sharey=True, layout='tight')
fig.subplots_adjust(wspace=0.2, hspace=0.2)
## Positive frequencies
spectrums[0, 0, -1] = 1.0
spectrums[0, 0, 2] = 0.0
particle = np.fft.ifft(spectrums[0, 0, :] * nfreq)
axs[0][0].plot(particle.real, particle.imag, 'C2.-', label=r'$ A_{-1} = 1.0 $')
#axs[0][0].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[0][0].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[0][0].set_box_aspect(1.0)
axs[0][0].set_aspect('equal')
axs[0][0].legend(loc='center', fontsize=fs)
#axs[0][0].set_title(r'n = -1', fontsize=14)
axs[0][0].set_xlim([-1.5, 1.5])
axs[0][0].set_ylim([-1.5, 1.5])
axs[0][0].set_xticks([-1, 0, 1])
axs[0][0].set_yticks([-1, 0, 1])

spectrums[0, 1, -1] = 1.0
spectrums[0, 1, 1] = 0.2*1j
particle = np.fft.ifft(spectrums[0, 1, :] * nfreq)
axs[0][1].plot(particle.real, particle.imag, 'C2.-', label=r'$ A_{1} = 0.2i $')
#axs[0][1].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[0][1].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[0][1].set_aspect('equal')
axs[0][1].set_box_aspect(1.0)
axs[0][1].legend(loc='center', fontsize=fs)
#axs[0][1].set_title(r'n = -1', fontsize=14)

spectrums[0, 2, -1] = 1.0
spectrums[0, 2, 2] = 0.2
particle = np.fft.ifft(spectrums[0, 2, :] * nfreq)
axs[0][2].plot(particle.real, particle.imag, 'C2.-', label=r'$ A_{2} = 0.2 $')
#axs[0][2].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[0][2].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[0][2].set_box_aspect(1.0)
axs[0][2].set_aspect('equal')
axs[0][2].legend(loc='center', fontsize=fs)
#fig.suptitle(r'Fourier descriptors', fontsize=14)

spectrums[0, 3, -1] = 1.0
spectrums[0, 3, 3] = 0.1*1j
particle = np.fft.ifft(spectrums[0, 3, :] * nfreq)
axs[0][3].plot(particle.real, particle.imag, 'C2.-', label=r'$ A_{3} = 0.1i $')
#axs[0][3].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[0][3].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[0][3].set_box_aspect(1.0)
axs[0][3].set_aspect('equal')
axs[0][3].legend(loc='center',fontsize=fs)
#axs[0][3].set_title(r'n = 3', fontsize=14)

spectrums[0, 4, -1] = 1.0
spectrums[0, 4, 4] = 0.1
particle = np.fft.ifft(spectrums[0, 4, :] * nfreq)
axs[0][4].plot(particle.real, particle.imag, 'C2.-', label=r'$ A_{4} = 0.1 $')
#axs[0][4].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[0][4].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[0][4].set_box_aspect(1.0)
axs[0][4].set_aspect('equal')
axs[0][4].legend(loc='center', fontsize=fs)
#axs[0][4].set_title(r'n = 4', fontsize=14)

# spectrums[0, 5, -1] = 1.0
# spectrums[0, 5, 5] = 0.1
# particle = np.fft.ifft(spectrums[0, 5, :] * nfreq)
# axs[0][5].plot(particle.real, particle.imag, 'r.-', label=r'$k = 5$')
# axs[0][5].axhline(y = 0.0, c='k', ls='-', lw=0.5)
# axs[0][5].axvline(x = 0.0, c='k', ls='-', lw=0.5)
# axs[0][5].set_aspect('equal')
# axs[0][5].legend(loc='center')

## Negative frequencies
spectrums[1, 0, -1] = 1.0
spectrums[1, 0, -2] = 0.3
particle = np.fft.ifft(spectrums[1, 0, :] * nfreq)
axs[1][0].plot(particle.real, particle.imag, 'C0.-', label=r'$ A_{-2} = 0.3 $')
#axs[1][0].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[1][0].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[1][0].set_box_aspect(1.0)
axs[1][0].set_aspect('equal')
axs[1][0].legend(loc='center', fontsize=fs)
#axs[1][0].set_title(r'n = -2', fontsize=14)

spectrums[1, 1, -1] = 1.0
spectrums[1, 1, -3] = -0.2*1j
particle = np.fft.ifft(spectrums[1, 1, :] * nfreq)
axs[1][1].plot(particle.real, particle.imag, 'C0.-', label=r'$ A_{-3} = -0.2i $')
#axs[1][1].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[1][1].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[1][1].set_box_aspect(1.0)
axs[1][1].set_aspect('equal')
axs[1][1].legend(loc='center', fontsize=fs)
#axs[1][1].set_title(r'n = -3', fontsize=14)

spectrums[1, 2, -1] = 1.0
spectrums[1, 2, -4] = -0.2
particle = np.fft.ifft(spectrums[1, 2, :] * nfreq)
axs[1][2].plot(particle.real, particle.imag, 'C0.-', label=r'$ A_{-4} = -0.2 $')
#axs[1][2].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[1][2].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[1][2].set_box_aspect(1.0)
axs[1][2].set_aspect('equal')
axs[1][2].legend(loc='center', fontsize=fs)
#axs[1][2].set_title(r'n = -4', fontsize=14)

spectrums[1, 3, -1] = 1.0
spectrums[1, 3, -5] = -0.1*1j
particle = np.fft.ifft(spectrums[1, 3, :] * nfreq)
axs[1][3].plot(particle.real, particle.imag, 'C0.-', label=r'$ A_{-5} = -0.1i $')
#axs[1][3].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[1][3].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[1][3].set_box_aspect(1.0)
axs[1][3].set_aspect('equal')
axs[1][3].legend(loc='center',fontsize=fs)
# axs[1][3].set_title(r'n = -5', fontsize=14)

spectrums[1, 4, -1] = 1.0
spectrums[1, 4, -6] = 0.1
particle = np.fft.ifft(spectrums[1, 4, :] * nfreq)
axs[1][4].plot(particle.real, particle.imag, 'C0.-', label=r'$ A_{-6} = 0.1 $')
#axs[1][4].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[1][4].axvline(x = 0.0, c='k', ls='-', lw=0.5)
axs[1][4].set_box_aspect(1.0)
axs[1][4].set_aspect('equal')
axs[1][4].legend(loc='center', fontsize=fs)
#axs[1][1].set_title(r'n = -6', fontsize=14)

# spectrums[1, 5, -1] = 1.0
# spectrums[1, 5, -7] = 0.1
# particle = np.fft.ifft(spectrums[1, 5, :] * nfreq)
#axs[1][5].plot(particle.real, particle.imag, 'C0.-', label=r'$k = -7$')
#axs[1][5].axhline(y = 0.0, c='k', ls='-', lw=0.5)
#axs[1][5].axvline(x = 0.0, c='k', ls='-', lw=0.5)
#axs[1][5].set_aspect('equal')
#axs[1][5].set_title(r'k = -7')
#axs[1][5].legend(loc='center')

plt.minorticks_off()
plt.savefig('img\spectrums.svg', dpi=330, transparent=False)
plt.show()
