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

    def generate_by_amplitude(self, A1=0.121, A3=0.084, A16=0.0039, A37=0.0011):
        # normalized frequencies
        self.spectrum[-1] = A1      # A1 < 0.5     
        self.spectrum[0] = 0.0      
        self.spectrum[1] = 1.0       
        self.spectrum[2] = 0.0
        # positive frequencies     
        self.spectrum[3] = A3       # A3 < 0.15
        self.spectrum[4:16] = np.power(2.0, -2.0*np.log2(np.arange(4, 16)/(3)) + np.log2(A3)) if A3 > 0 else 0.0
        self.spectrum[16] = A16     # A16 < 0.01
        self.spectrum[17:37] = np.power(2.0, -2.0*np.log2(np.arange(17, 37)/(16)) + np.log2(A16)) if A16 > 0 else 0.0
        self.spectrum[37] = A37     # A37 < 0.002
        self.spectrum[38:64] = np.power(2.0, -2.0*np.log2(np.arange(38, 64)/(37)) + np.log2(A37)) if A37 > 0 else 0.0
        # negative frequencies
        self.spectrum[-2] = A3
        self.spectrum[-3:-15:-1] = np.power(2.0, -2.0*np.log2(np.arange(-3, -15, -1)/(-2)) + np.log2(A3)) if A3 > 0 else 0.0
        self.spectrum[-15] = A16
        self.spectrum[-16:-36:-1] = np.power(2.0, -2.0*np.log2(np.arange(-16, -36, -1)/(-15)) + np.log2(A16)) if A16 > 0 else 0.0
        self.spectrum[-36] = A37
        self.spectrum[-37:-64:-1] = np.power(2.0, -2.0*np.log2(np.arange(-37, -64, -1)/(-36)) + np.log2(A37)) if A37 > 0 else 0.0

        while True:
            nfreqs = len(self.spectrum)
            phases = np.exp(1j*np.random.normal(-np.pi, np.pi, nfreqs))
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
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Arial']
    matplotlib.rcParams['font.size'] = 12

    fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained')
    axs[0].set_xlim([-2.0, 2.0])
    axs[0].set_ylim([-2.0, 2.0])
    axs[0].axhline(y=0.0, c='b', lw=1.0, ls='--')
    axs[0].axvline(x=0.0, c='b', lw=1.0, ls='--')
    axs[0].set_box_aspect(1.0)

    #np.random.seed(1)
    particle_factory = FFTGenerator(nfreq=128)
    particle = particle_factory.generate_by_amplitude()
    # characterization
    print(f'Particle Area (A): {particle.calc_area()}')
    print(f'Particle Perimeter (P): {particle.calc_perimeter()}')
    print(f'Particle Elongation Index (EI): {particle.calc_elongation()}')
    print(f'Particle Roundness Index (RI): {particle.calc_roundness()}')
    print(f'Particle Angularity Index (AI): {particle.calc_angularity()}')
    print(particle.is_within_domain((-2.0, -2.0, 2.0, 2.0), tol=0.0))
    # visualization
    particle.render(axs[0], 'cyan', add_rect=True)

    x = particle.points[:, 0]
    y = particle.points[:, 1]
    freq = np.fft.fft(x + y*1j, norm='forward')
    axs[1].bar(np.fft.fftfreq(128, 1.0/128), np.abs(freq))
    #axs[1].set_box_aspect(1.0)
    #axs[1].set_xlim([2, 50])

    fig.savefig('shape_descriptor.svg')
    plt.show()
