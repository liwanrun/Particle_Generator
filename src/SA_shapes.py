import numpy as np
from particle import PolygonParticle

np.random.seed()
phases = np.exp(1j*np.random.uniform(-np.pi, np.pi, 128))
#np.random.seed()

def generate_paticle_IDFT(amp):
        A1, A3, A16, A37 = amp
        spectrum = np.zeros(128)
        # normalized frequencies
        spectrum[-2] = 0.0
        spectrum[-1] = 1.0     # A1 < 0.5        
        spectrum[0] = 0.0      
        spectrum[1] = A1         
        # positive frequencies     
        spectrum[2] = A3       # A3 < 0.15
        spectrum[3:15] = np.power(2.0, -2.0*np.log2(np.arange(3, 15)/(2)) + np.log2(A3)) if A3 > 0 else 0.0
        spectrum[15] = A16     # A16 < 0.01
        spectrum[16:36] = np.power(2.0, -2.0*np.log2(np.arange(16, 36)/(15)) + np.log2(A16)) if A16 > 0 else 0.0
        spectrum[36] = A37     # A37 < 0.002
        spectrum[37:64] = np.power(2.0, -2.0*np.log2(np.arange(37, 64)/(36)) + np.log2(A37)) if A37 > 0 else 0.0
        # negative frequencies
        spectrum[-3] = A3
        spectrum[-4:-16:-1] = np.power(2.0, -2.0*np.log2(np.arange(-4, -16, -1)/(-3)) + np.log2(A3)) if A3 > 0 else 0.0
        spectrum[-16] = A16
        spectrum[-17:-37:-1] = np.power(2.0, -2.0*np.log2(np.arange(-17, -37, -1)/(-16)) + np.log2(A16)) if A16 > 0 else 0.0
        spectrum[-37] = A37
        spectrum[-38:-65:-1] = np.power(2.0, -2.0*np.log2(np.arange(-38, -65, -1)/(-37)) + np.log2(A37)) if A37 > 0 else 0.0

        nfreqs = len(spectrum)
        signal = np.fft.ifft(nfreqs * spectrum * phases)
        points = np.vstack((signal.real, signal.imag)).T
        particle = PolygonParticle(points)
        return particle

def target_function(x):
    particle = generate_paticle_IDFT(x)
    if particle.is_valid():
        active_shapes = particle.calc_shape_indexes()
        target_shapes = np.array([1.6798, 0.8232, 3.6584])
        print(active_shapes)
        return np.dot(((active_shapes[:3] - target_shapes[:3])**2), np.array([1/0.1, 1/0.1, 1/0.1]))
    else:
         return 1.0e+100


if __name__ == '__main__':

    from sko.SA import SAFast
    import matplotlib.pyplot as plt

    sa_fast = SAFast(func=target_function, x0=[0.45, 0.075, 0.005, 0.0005], T_max=100, T_min=1e-9, quench=0.99, L=300, max_stay_counter=150,
                 lb=[0.00, 0.00, 0.00, 0.00], ub=[0.5, 0.15, 0.01, 0.001])
    sa_fast.run()
    print('***Fast Simulated Annealing with bounds best_x is ', sa_fast.best_x, 'best_y is ', sa_fast.best_y)

    fig, ax = plt.subplots(nrows=1, ncols=2, layout='constrained')
    ax[0].plot(sa_fast.generation_best_Y)
    ax[0].set_title(f'solution: {sa_fast.best_y}')
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Error")
    ax[0].set_aspect(1.0)
    ax[0].set_box_aspect(1.0)

    particle = generate_paticle_IDFT(sa_fast.best_x)
    particle.render(ax[1], 'cyan', add_rect=True)
    ax[1].set_aspect(1.0)
    ax[1].set_box_aspect(1.0)
    print(f'Particle Shape Indexes: {particle.calc_shape_indexes()}')
    fig.savefig('SA.svg', transparent=True)
    plt.show()
