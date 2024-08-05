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

    def generate_by_amplitude(self, A1=0.150, A3=0.075, A16=0.005, A37=0.001):
        # normalized frequencies
        self.spectrum[-1] = A1      # A1 < 0.5
        self.spectrum[0] = 0.0      
        self.spectrum[1] = 1.0
        self.spectrum[2] = 0.0
        # positive frequencies     
        self.spectrum[3] = A3       # A3 < 0.15
        self.spectrum[4:16] = np.power(2.0, -2.0*np.log2(np.arange(4, 16)/(3)) + np.log2(A3))
        self.spectrum[16] = A16     # A16 < 0.01
        self.spectrum[17:37] = np.power(2.0, -2.0*np.log2(np.arange(17, 37)/(16)) + np.log2(A16))
        self.spectrum[37] = A37     # A37 < 0.002
        self.spectrum[38:64] = np.power(2.0, -2.0*np.log2(np.arange(38, 64)/(37)) + np.log2(A37))
        # negative frequencies
        self.spectrum[-2] = A3
        self.spectrum[-3:-15] = np.power(2.0, -2.0*np.log2(np.arange(-3, -15)/(-2)) + np.log2(A3))
        self.spectrum[-15] = A16
        self.spectrum[-16:-36] = np.power(2.0, -2.0*np.log2(np.arange(-16, -36)/(-15)) + np.log2(A16))
        self.spectrum[-36] = A37
        self.spectrum[-37:-64] = np.power(2.0, -2.0*np.log2(np.arange(-37, -64)/(-36)) + np.log2(A37))

        nfreqs = len(self.spectrum)
        phases = np.exp(1j*np.random.normal(-np.pi, np.pi, nfreqs))
        signal = np.fft.ifft(nfreqs * self.spectrum * phases)
        points = np.vstack((signal.real, signal.imag)).T
        return points

    def generate_by_mophology(self, EI, RI, AI):
        pass

## Debug ##
if __name__ == '__main__':
    fig, axs = plt.subplots()
    axs.set_xlim([-2.0, 2.0])
    axs.set_ylim([-2.0, 2.0])
    axs.set_aspect('equal')

    # np.random.seed(7)
    particle_factory = FFTGenerator(nfreq=128)
    coords = particle_factory.generate_by_amplitude()
    # square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.7, 1.0], [0.7, 0.5], [0.3, 0.5], [0.3, 1.0], [0.0, 1.0]])
    particle = PolygonParticle(coords)

    if particle.is_valid():
        # characterization
        print(f'Particle Area (A): {particle.calc_area()}')
        print(f'Particle Perimeter (P): {particle.calc_perimeter()}')
        print(f'Particle Elongation Index (EI): {particle.calc_elongation()}')
        print(f'Particle Roundness Index (RI): {particle.calc_roundness()}')
        print(f'Particle Angularity Index (AI): {particle.calc_angularity()}')
        # visualization
        particle.render(axs)
        plt.show()
    else:
        print('This particle is not valid!')

