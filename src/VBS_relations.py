from particle import PolygonParticle
import numpy as np

class RegressionFitting:
     def __init__(self) -> None:
          pass
     
     def generate_particle(self, A2, A3, A16, A37) -> PolygonParticle:
          amplitudes = np.zeros(128, dtype=np.float64)
          amplitudes[-2] = 0.0
          amplitudes[-1] = 1.0       
          amplitudes[0] = 0.0 
          amplitudes[1] = A2       # A2 < 0.5     
          # positive frequencies     
          amplitudes[2] = A3       # A3 < 0.15
          amplitudes[3:15] = np.power(2.0, -2.0*np.log2(np.arange(3, 15)/(2)) + np.log2(A3)) if A3 > 0.0 else 0.0
          amplitudes[15] = A16     # A16 < 0.01
          amplitudes[16:36] = np.power(2.0, -2.0*np.log2(np.arange(16, 36)/(15)) + np.log2(A16)) if A16 > 0.0 else 0.0
          amplitudes[36] = A37     # A37 < 0.002
          amplitudes[37:64] = np.power(2.0, -2.0*np.log2(np.arange(37, 64)/(36)) + np.log2(A37)) if A37 > 0.0 else 0.0
          # negative frequencies
          amplitudes[-3] = A3
          amplitudes[-4:-16:-1] = np.power(2.0, -2.0*np.log2(np.arange(-4, -16, -1)/(-3)) + np.log2(A3)) if A3 > 0.0 else 0.0
          amplitudes[-16] = A16
          amplitudes[-17:-37:-1] = np.power(2.0, -2.0*np.log2(np.arange(-17, -37, -1)/(-16)) + np.log2(A16)) if A16 > 0.0 else 0.0
          amplitudes[-37] = A37
          amplitudes[-38:-65:-1] = np.power(2.0, -2.0*np.log2(np.arange(-38, -65, -1)/(-37)) + np.log2(A37)) if A37 > 0.0 else 0.0

          while True:
            nfreqs = len(amplitudes)
            phases = np.exp(1j*np.random.uniform(0.0, 2*np.pi, nfreqs))
            signal = np.fft.ifft(nfreqs * amplitudes * phases)
            points = np.vstack((signal.real, signal.imag)).T
            particle = PolygonParticle(points)
            if particle.is_valid(): 
                return particle 

     def EI14VersusA2(self):
        A2_series = np.linspace(0.01, 0.50, 50)
        EI_fitted = 5.3738*np.power(A2_series, 1.4955) + 1.0
        A2_array = []; EI_array = []
        for A2 in A2_series:
            EI_sum = 0.0
            for _ in np.arange(50):
                A3 = np.random.uniform(0.0, 0.15)
                A16 = np.random.uniform(0.0, 0.01)
                A37 = np.random.uniform(0.0, 0.002)
                p = self.generate_particle(A2, A3, A16, A37)
                EI, _, _ = p.calc_shape_indexes()
                EI_sum += EI
            A2_array.append(A2)
            EI_array.append(EI_sum/50)
        return [A2_array, EI_array, A2_series, EI_fitted]
         
     def RD12VersusA3(self):
         A3_series = np.linspace(0.003, 0.15, 50)
         RD_fitted = -6.7031*np.power(A3_series, 1.9438)
         A3_array = []; RD_array = []
         for A3 in A3_series:
             for _ in np.arange(60):
                 A2 = np.random.uniform(0.0, 0.5)
                 p1 = self.generate_particle(A2, A3=0.0, A16=0.0, A37=0.0)
                 _, RI_1, _ = p1.calc_shape_indexes()
                 p2 = self.generate_particle(A2, A3, A16=0.0, A37=0.0)
                 _, RI_2, _ = p2.calc_shape_indexes()
                 A3_array.append(A3)
                 RD_array.append(RI_2 - RI_1)
         return [A3_array, RD_array, A3_series, RD_fitted]
     
     def RD23VersusA16(self):
         A16_series = np.linspace(0.0002, 0.01, 50)
         RD_fitted = -7.7494*np.power(10.0*A16_series, 1.9558)
         A16_array = []; RD_array = []
         for A16 in A16_series:
             for _ in np.arange(60):
                 A2 = 0.25
                 A3 = np.random.uniform(0.0, 0.15)
                 p1 = self.generate_particle(A2, A3, A16=0.0, A37=0.0)
                 _, RI_1, _ = p1.calc_shape_indexes()
                 p2 = self.generate_particle(A2, A3, A16, A37=0.0)
                 _, RI_2, _ = p2.calc_shape_indexes()
                 A16_array.append(A16)
                 RD_array.append(RI_2 - RI_1)
         return [A16_array, RD_array, A16_series, RD_fitted]
     
     def RD34VersusA37(self):
         A37_series = np.linspace(0.0004, 0.002, 50)
         RD_fitted = -44.9054*np.power(10.0*A37_series, 1.9432)
         A37_array = []; RD_array = []
         for A37 in A37_series:
             for _ in np.arange(60):
                 A2 = 0.25
                 A3 = np.random.uniform(0.0, 0.15)
                 A16 = np.random.uniform(0.0, 0.01)
                 # np.random.seed(10)
                 p1 = self.generate_particle(A2, A3, A16, A37=0.0)
                 _, RI_1, _ = p1.calc_shape_indexes()
                 p2 = self.generate_particle(A2, A3, A16, A37)
                 _, RI_2, _ = p2.calc_shape_indexes()
                 np.random.seed()
                 A37_array.append(A37)
                 RD_array.append(RI_2 - RI_1)
         return [A37_array, RD_array, A37_series, RD_fitted]
     
     def AD34VersusA37(self):
         A37_series = np.linspace(0.0004, 0.002, 50)
         AD_fitted = 1.4454*np.power(1000.0*A37_series, 1.6076)
         A37_array = []; AD_array = []
         for A37 in A37_series:
             for _ in np.arange(60):
                 A2 = 0.25
                 A3 = np.random.uniform(0.0, 0.15)
                 A16 = np.random.uniform(0.0, 0.01)
                 # np.random.seed(10)
                 p1 = self.generate_particle(A2, A3, A16, A37=0.0)
                 _, _, AI_1 = p1.calc_shape_indexes()
                 p2 = self.generate_particle(A2, A3, A16, A37)
                 _, _, AI_2 = p2.calc_shape_indexes()
                 np.random.seed()
                 A37_array.append(A37)
                 AD_array.append(AI_2 - AI_1)
         return [A37_array, AD_array, A37_series, AD_fitted]
     

if __name__ == '__main__':
    # Regression
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif=['Times'])
    #plt.style.use('ggplot')
    plt.style.use('seaborn-v0_8')

    fitter = RegressionFitting()
    #A2, EI, A2_fit, EI_fit = fitter.EI14VersusA2()
    #A3, RD, A3_fit, RD_fit = fitter.RD12VersusA3() 
    #A16, RD, A16_fit, RD_fit = fitter.RD23VersusA16()
    A37, AD, A37_fit, AD_fit = fitter.AD34VersusA37()

    fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax.plot(A37, AD, '.', label=r'Random')
    ax.plot(A37_fit, AD_fit, 'k-', lw=2.0, label=r'Fitted')
    #ax.set_xlim([0.0, 0.15])
    #ax.set_ylim([-0.15, 0.025])
    ax.set_xlabel(r'$A_{16}$')
    ax.set_ylabel(r'$RD_{2-3}$')
    ax.legend()
    plt.show()