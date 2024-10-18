# Visiualization of the relationship between Fourier descriptors and particle morphology
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Calibri']
matplotlib.rcParams['font.size'] = 10.5

nfreq = 128
spectrums = np.zeros((2, 6, nfreq), dtype='complex')

fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(18, 6), sharex=True, sharey=True, layout='constrained')
## Positive frequencies
spectrums[0, 0, 1] = 1.0
# spectrums[0, 0, 0] = 0.3
particle = np.fft.ifft(spectrums[0, 0, :] * nfreq)
axs[0][0].plot(particle.real, particle.imag, 'r.-', label='k = -1')
axs[0][0].set_box_aspect(1.0)
axs[0][0].set_xlim([-1.5, 1.5])
axs[0][0].set_ylim([-1.5, 1.5])
axs[0][0].axhline(y = 0.0, c='k', ls=':')
axs[0][0].axvline(x = 0.0, c='k', ls=':')
axs[0][0].legend()

spectrums[0, 1, -1] = 1.0
spectrums[0, 1, 1] = 0.3
particle = np.fft.ifft(spectrums[0, 1, :] * nfreq)
axs[0][1].plot(particle.real, particle.imag, 'r.-', label='k = 1')
axs[0][1].set_aspect('equal')
axs[0][1].axhline(y = 0.0, c='k', ls=':')
axs[0][1].axvline(x = 0.0, c='k', ls=':')
axs[0][1].legend()

spectrums[0, 2, -1] = 1.0
spectrums[0, 2, 2] = 0.4
particle = np.fft.ifft(spectrums[0, 2, :] * nfreq)
axs[0][2].plot(particle.real, particle.imag, 'r.-', label='k = 2')
axs[0][2].set_aspect('equal')
axs[0][2].axhline(y = 0.0, c='k', ls=':')
axs[0][2].axvline(x = 0.0, c='k', ls=':')
axs[0][2].legend()

spectrums[0, 3, -1] = 1.0
spectrums[0, 3, 3] = -0.114
particle = np.fft.ifft(spectrums[0, 3, :] * nfreq)
axs[0][3].plot(particle.real, particle.imag, 'r.-', label='k = 3')
axs[0][3].set_aspect('equal')
axs[0][3].axhline(y = 0.0, c='k', ls=':')
axs[0][3].axvline(x = 0.0, c='k', ls=':')
axs[0][3].legend()

spectrums[0, 4, -1] = 1.0
spectrums[0, 4, 4] = 0.2
particle = np.fft.ifft(spectrums[0, 4, :] * nfreq)
axs[0][4].plot(particle.real, particle.imag, 'r.-', label='k = 4')
axs[0][4].set_aspect('equal')
axs[0][4].axhline(y = 0.0, c='k', ls=':')
axs[0][4].axvline(x = 0.0, c='k', ls=':')
axs[0][4].legend()

spectrums[0, 5, -1] = 1.0
spectrums[0, 5, 5] = 0.2
particle = np.fft.ifft(spectrums[0, 5, :] * nfreq)
axs[0][5].plot(particle.real, particle.imag, 'r.-', label='k = 5')
axs[0][5].set_aspect('equal')
axs[0][5].axhline(y = 0.0, c='k', ls=':')
axs[0][5].axvline(x = 0.0, c='k', ls=':')
axs[0][5].legend()

## Negative frequencies
spectrums[1, 0, -1] = 1.0
spectrums[1, 0, -2] = 0.3
particle = np.fft.ifft(spectrums[1, 0, :] * nfreq)
axs[1][0].plot(particle.real, particle.imag, 'b.-', label='k = -2')
axs[1][0].set_aspect('equal')
axs[1][0].axhline(y = 0.0, c='k', ls=':')
axs[1][0].axvline(x = 0.0, c='k', ls=':')
axs[1][0].legend()

spectrums[1, 1, -1] = 1.0
spectrums[1, 1, -3] = 0.3
particle = np.fft.ifft(spectrums[1, 1, :] * nfreq)
axs[1][1].plot(particle.real, particle.imag, 'b.-', label='k = -3')
axs[1][1].set_aspect('equal')
axs[1][1].axhline(y = 0.0, c='k', ls=':')
axs[1][1].axvline(x = 0.0, c='k', ls=':')
axs[1][1].legend()

spectrums[1, 2, -1] = 1.0
spectrums[1, 2, -4] = 0.2
particle = np.fft.ifft(spectrums[1, 2, :] * nfreq)
axs[1][2].plot(particle.real, particle.imag, 'b.-', label='k = -4')
axs[1][2].set_aspect('equal')
axs[1][2].axhline(y = 0.0, c='k', ls=':')
axs[1][2].axvline(x = 0.0, c='k', ls=':')
axs[1][2].legend()

spectrums[1, 3, -1] = 1.0
spectrums[1, 3, -5] = 0.2
particle = np.fft.ifft(spectrums[1, 3, :] * nfreq)
axs[1][3].plot(particle.real, particle.imag, 'b.-', label='k = -5')
axs[1][3].set_aspect('equal')
axs[1][3].axhline(y = 0.0, c='k', ls=':')
axs[1][3].axvline(x = 0.0, c='k', ls=':')
axs[1][3].legend()

spectrums[1, 4, -1] = 1.0
spectrums[1, 4, -6] = 0.1
particle = np.fft.ifft(spectrums[1, 4, :] * nfreq)
axs[1][4].plot(particle.real, particle.imag, 'b.-', label='k = -6')
axs[1][4].set_aspect('equal')
axs[1][4].axhline(y = 0.0, c='k', ls=':')
axs[1][4].axvline(x = 0.0, c='k', ls=':')
axs[1][4].legend()

spectrums[1, 5, -1] = 1.0
spectrums[1, 5, -7] = 0.1
particle = np.fft.ifft(spectrums[1, 5, :] * nfreq)
axs[1][5].plot(particle.real, particle.imag, 'b.-', label='k = -7')
axs[1][5].set_aspect('equal')
axs[1][5].axhline(y = 0.0, c='k', ls=':')
axs[1][5].axvline(x = 0.0, c='k', ls=':')
axs[1][5].legend()

plt.minorticks_off()
plt.savefig('img\spectrums.svg', dpi=330, transparent=True)
plt.show()
