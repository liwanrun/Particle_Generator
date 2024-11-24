import numpy as np
import matplotlib.pyplot as plt
from particle import *
import rich

class FFTGenerator:
    '''Irregular particle generator with complex FFT
       Elongation(EI); Roundness(RI); Angularity(AI)
    '''
    def __init__(self, nfreq=128) -> None:
        self.spectrum = np.zeros(nfreq, dtype='complex')

    def generate_by_amplitude(self, A1=0.25, A3=0.075, A16=0.005, A37=0.0001):
        # normalized frequencies
        self.spectrum[-2] = 0.0
        self.spectrum[-1] = 1.0          
        self.spectrum[0] = 0.0      
        self.spectrum[1] = A1       # A1 < 0.5      
        # positive frequencies     
        self.spectrum[2] = A3       # A3 < 0.15
        self.spectrum[3:15] = np.power(2.0, -2.0*np.log2(np.arange(3, 15)/(2)) + np.log2(A3)) if A3 > 0.0 else 0.0
        self.spectrum[15] = A16     # A16 < 0.01
        self.spectrum[16:36] = np.power(2.0, -2.0*np.log2(np.arange(16, 36)/(15)) + np.log2(A16)) if A16 > 0.0 else 0.0
        self.spectrum[36] = A37     # A37 < 0.002
        self.spectrum[37:64] = np.power(2.0, -2.0*np.log2(np.arange(37, 64)/(36)) + np.log2(A37)) if A37 > 0.0 else 0.0
        # negative frequencies
        self.spectrum[-3] = A3
        self.spectrum[-4:-16:-1] = np.power(2.0, -2.0*np.log2(np.arange(-4, -16, -1)/(-3)) + np.log2(A3)) if A3 > 0.0 else 0.0
        self.spectrum[-16] = A16
        self.spectrum[-17:-37:-1] = np.power(2.0, -2.0*np.log2(np.arange(-17, -37, -1)/(-16)) + np.log2(A16)) if A16 > 0.0 else 0.0
        self.spectrum[-37] = A37
        self.spectrum[-38:-65:-1] = np.power(2.0, -2.0*np.log2(np.arange(-38, -65, -1)/(-37)) + np.log2(A37)) if A37 > 0.0 else 0.0

        rng = np.random.default_rng()
        while True:
            nfreqs = len(self.spectrum)
            phases = np.exp(1j*rng.uniform(-np.pi, np.pi, 128))
            signal = np.fft.ifft(nfreqs * self.spectrum * phases)
            points = np.vstack((signal.real, signal.imag)).T
            particle = PolygonParticle(points)
            if particle.is_valid(): 
                return particle

    def generate_by_mophology(self, EI, RI, AI):
        pass

## Debug ##
if __name__ == '__main__':
    import matplotlib
    np.seterr(divide='ignore')
 #   matplotlib.rcParams['font.family'] = 'serif'
 #   matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['font.size'] = 14

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif=['Times'])
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    fig, ax1 = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax1.set_xlim([-2.0, 2.0])
    ax1.set_ylim([-2.0, 2.0])
    ax1.axhline(y=0.0, c='b', lw=1.0, ls='--')
    ax1.axvline(x=0.0, c='b', lw=1.0, ls='--')
    ax1.set_aspect(1.0)
    ax1.set_box_aspect(1.0)

    np.random.seed(2)
    num_freqs = 128
    particle_factory = FFTGenerator(nfreq=num_freqs)
    #A2, A3, A16, A37 = 0.25, 0.075, 0.005, 0.0005
    A2, A3, A16, A37 = 0.1878819,  0.09034526, 0.00451293, 0.0006388
    # A2, A3, A16, A37 = 0.445709,   0.0,  0.0, 0.0
    particle = particle_factory.generate_by_amplitude(A2, A3, A16, A37)
    np.random.seed()
    # characterization
    print(f'Particle Area (A): {particle.calc_area()}')
    print(f'Particle Perimeter (P): {particle.calc_perimeter()}')
    print(f'Particle Elongation Index (EI): {particle.calc_elongation()}')
    print(f'Particle Roundness Index (RI): {particle.calc_roundness()}')
    print(f'Particle Angularity Index (AI): {particle.calc_angularity()}')
    print(particle.is_closely_within_domain((-2.0, -2.0, 2.0, 2.0), tol=0.0))
    print(f'Particle Shape Indexes: {particle.calc_shape_indexes()}')
    # visualization
    particle.render(ax1, 'cyan', add_rect=True)

    fig.savefig('A2.svg')
 #   plt.show()

    ## Frequency-Maginitude
    f, a = plt.subplots(nrows=4, ncols=1, figsize=(10, 8), sharex=False, layout='constrained')
    x = particle.points[:, 0]
    y = particle.points[:, 1]
    freq = np.fft.fft(x + y*1j, norm='forward')
    a[0].bar(np.fft.fftfreq(num_freqs, 1.0/num_freqs), np.abs(freq))
    #ax2.set_box_aspect(1.0)
    a[0].set_xlim([-64, 63])
    a[0].set_ylim([0.0, 1.0])
    a[0].set_ylabel('Magnitude')
    a[0].set_xticks([-64, -37, -16, -3, 2, 15, 36, 63])
    a[0].set_yticks(np.arange(0.0, 1.2, 0.20))
    a[0].text(x=50, y=0.5, s=r'\begin{align*} A_{-2} &= 0.0 \\ A_{-1} &= 1.0 \\ A_{0} &= 0.0 \\ A_{1} &= 0.25 \end{align*',
              horizontalalignment='right', verticalalignment='center')

    a[1].bar(np.fft.fftfreq(num_freqs, 1.0/num_freqs), np.abs(freq))
    #ax2.set_box_aspect(1.0)
    a[1].set_xlim([-64, 63])
    a[1].set_ylim([0.0, 0.1])
 #   a[1].set_xlabel('Frequency')
    a[1].set_ylabel('Magnitude')
    a[1].set_xticks([-64, -37, -16, -3, 2, 15, 36, 63])
    a[1].set_yticks(np.linspace(0.0, 0.10, 6))
    a[1].ticklabel_format(style='sci', axis='y', scilimits=(0,1))
    x2 = np.arange(2, 15); y2 = np.power(2.0, -2.0*np.log2(np.arange(2, 15)/(2)) + np.log2(A3)); y2[0] = A3
    x_3 = np.arange(-3, -16, -1); y_3 = np.power(2.0, -2.0*np.log2(np.arange(-3, -16, -1)/(-3)) + np.log2(A3)); y_3[0] = A3
    a[1].plot(x2, y2, 'r', lw=2.0, label=r'$ A_{n} = 2^{-2 \times log_{2} (n/2) + log_{2}A_{2}} $')
    a[1].plot(x_3, y_3, 'k', lw=2.0, label=r'$ A_{n} = 2^{-2 \times log_{2} (n/-3) + log_{2}A_{-3}} $')
    a[1].legend(frameon=False)

    a[2].bar(np.fft.fftfreq(num_freqs, 1.0/num_freqs), np.abs(freq))
    a[2].set_xlim([-64, 63])
    a[2].set_ylim([0.0, 0.01])
#    a[2].set_xlabel('Frequency')
    a[2].set_ylabel('Magnitude')
    a[2].set_xticks([-64, -37, -16, -3, 2, 15, 36, 63])
    a[2].set_yticks(np.linspace(0.0, 0.010, 6))
    a[2].ticklabel_format(style='sci', axis='y', scilimits=(0,1))
    x15 = np.arange(15, 36); y15 = np.power(2.0, -2.0*np.log2(np.arange(15, 36)/(15)) + np.log2(A16)); y15[0] = A16
    x_16 = np.arange(-16, -37, -1); y_16 = np.power(2.0, -2.0*np.log2(np.arange(-16, -37, -1)/(-16)) + np.log2(A16)); y_16[0] = A16
    a[2].plot(x15, y15, 'r', lw=2.0, label=r'$ A_{n} = 2^{-2 \times log_{2} (n/15) + log_{2}A_{15}} $')
    a[2].plot(x_16, y_16, 'k', lw=2.0, label=r'$ A_{n} = 2^{-2 \times log_{2} (n/-16) + log_{2}A_{-16}} $')
    a[2].legend(frameon=False)

    a[3].bar(np.fft.fftfreq(num_freqs, 1.0/num_freqs), np.abs(freq))
    a[3].set_xlim([-64, 63])
    a[3].set_ylim([0.0, 0.0015])
    a[3].set_xlabel('Frequency')
    a[3].set_ylabel('Magnitude')
    a[3].set_xticks([-64, -37, -16, -3, 2, 15, 36, 63])
    a[3].set_yticks(np.linspace(0.0, 0.0015, 6))
    a[3].ticklabel_format(style='sci', axis='y', scilimits=(0,1))
    x36 = np.arange(36, 64); y36 = np.power(2.0, -2.0*np.log2(np.arange(36, 64)/(36)) + np.log2(A37)); y36[0] = A37
    x_37 = np.arange(-37, -65, -1); y_37 = np.power(2.0, -2.0*np.log2(np.arange(-37, -65, -1)/(-37)) + np.log2(A37)); y_37[0] = A37
    a[3].plot(x36, y36, 'r', lw=2.0, label=r'$ A_{n} = 2^{-2 \times log_{2} (n/36) + log_{2}A_{36}} $')
    a[3].plot(x_37, y_37, 'k', lw=2.0, label=r'$ A_{n} = 2^{-2 \times log_{2} (n/-37) + log_{2}A_{-37}} $')
    a[3].legend(frameon=False, loc='upper right')

    f.savefig('frequency.svg', transparent=True)
    plt.show()
