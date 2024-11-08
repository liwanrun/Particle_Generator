## 
import numpy as np
import matplotlib.pyplot as plt
from particle import PolygonParticle

class VBSGenerator:
    '''Value By Stage method to generate particles given by target shape descriptors'''
    def __init__(self, EI, RI, AI) -> None:
        self.target_EI = EI
        self.target_RI = RI
        self.target_AI = AI
        self.EI_err = 1.0e-02
        self.RI_err = 1.0e-02
        self.AI_err = 1.0e-02
        self.particle = None
        self.phases = np.exp(1j*np.random.normal(0.0, np.pi, 128))
        self.amplitudes = np.zeros(128, dtype='float64')
        self.amplitudes[-2] = 0.0
        self.amplitudes[-1] = 1.0       
        self.amplitudes[0] = 0.0

    def IFFT_particle(self, stage:int) -> PolygonParticle:
        while True:
            '''Be careful dead loop if particle is invalid!'''
            if 1 == stage:
                self.update_phase_A2()
            elif 2 == stage:
                self.update_phase_A3()
            elif 3 == stage:
                self.update_phase_A16()
            elif 4 == stage:
                self.update_phase_A37()
            signal = np.fft.ifft(128 * self.amplitudes * self.phases)
            points = np.vstack((signal.real, signal.imag)).T
            particle = PolygonParticle(points)
            if particle.is_valid(): 
                return particle
            
    def update_A2(self) -> None:
        A2 = np.power((self.target_EI - 1.0) / 5.3738, (1.0 / 1.4955))
        self.amplitudes[1] = A2  

    def update_A3(self, RI) -> None:
        RD = 0.6 * (self.target_RI - RI)
        A3 = np.power((np.abs(RD) / 6.7031), (1.0 / 1.9438))
        self.amplitudes[2] = A3       # A3 < 0.15
        self.amplitudes[3:15] = np.power(2.0, -2.0*np.log2(np.arange(3, 15)/(2)) + np.log2(A3)) if A3 > 0.0 else 0.0
        self.amplitudes[-3] = A3
        self.amplitudes[-4:-16:-1] = np.power(2.0, -2.0*np.log2(np.arange(-4, -16, -1)/(-3)) + np.log2(A3)) if A3 > 0.0 else 0.0

    def update_A16(self, RI) -> None:
        RD = 0.8 * (self.target_RI - RI)
        A16 = np.power((np.abs(RD) / 7.7494), (1.0 / 1.9558)) / 10.0
        self.amplitudes[15] = A16     # A16 < 0.01
        self.amplitudes[16:36] = np.power(2.0, -2.0*np.log2(np.arange(16, 36)/(15)) + np.log2(A16)) if A16 > 0.0 else 0.0
        self.amplitudes[-16] = A16
        self.amplitudes[-17:-37:-1] = np.power(2.0, -2.0*np.log2(np.arange(-17, -37, -1)/(-16)) + np.log2(A16)) if A16 > 0.0 else 0.0

    def update_A37(self, A37) -> None:
        # AD = self.target_AI - AI
        # A37 = np.power((np.abs(AD) / 1.4454), (1.0 / 1.6076)) / 1000.0
        self.amplitudes[36] = A37     # A37 < 0.002
        self.amplitudes[37:64] = np.power(2.0, -2.0*np.log2(np.arange(37, 64)/(36)) + np.log2(A37)) if A37 > 0.0 else 0.0
        self.amplitudes[-37] = A37
        self.amplitudes[-38:-65:-1] = np.power(2.0, -2.0*np.log2(np.arange(-38, -65, -1)/(-37)) + np.log2(A37)) if A37 > 0.0 else 0.0

    
    def update_phase_A2(self) -> None:
        self.phases[1] = np.exp(1j*np.random.uniform(-np.pi/2, np.pi/2))
    
    def update_phase_A3(self) -> None:
        self.phases[2:15] = np.exp(1j*np.random.uniform(-np.pi, np.pi, 13))
        self.phases[-3:-16:-1] = np.exp(1j*np.random.uniform(-np.pi, np.pi, 13))
    
    def update_phase_A16(self) -> None:
        self.phases[15:36] = np.exp(1j*np.random.uniform(-np.pi, np.pi, 21))
        self.phases[-16:-37:-1] = np.exp(1j*np.random.uniform(-np.pi, np.pi, 21))
    
    def update_phase_A37(self) -> None:
        self.phases[36:64] = np.exp(1j*np.random.uniform(-np.pi, np.pi, 28))
        self.phases[-37:-65:-1] = np.exp(1j*np.random.uniform(-np.pi, np.pi, 28))

    ## Check criteria
    def check_stage_II(self, EI, RI) -> bool:
        ED = self.target_EI - EI
        RD = self.target_RI - RI
        cond_1 = (np.abs(ED) / self.target_EI) <= self.EI_err
        cond_2 = (RD <= 0.0) and (RD >= -0.08)
        return (cond_1 and cond_2)
    
    def check_stage_III_from_RI(self, EI, RI, AI) -> bool:
        ED = self.target_EI - EI
        RD = self.target_RI - RI
        AD = self.target_AI - AI
        cond_1 = (np.abs(ED) / self.target_EI) <= self.EI_err
        cond_2 = (RD <= 0.0)
        cond_3 = (AD >= 0.0) and (AD <= 4.5)
        return (cond_1 and cond_2 and cond_3)

    def check_stage_III_from_AI(self, RD) -> bool:
        return np.abs(RD) <= self.RI_err

    def check_stage_IV(self, EI, RI, AI) -> bool:
        ED = self.target_EI - EI
        RD = self.target_RI - RI
        AD = self.target_AI - AI
        cond_1 = (np.abs(ED) / self.target_EI) <= self.EI_err
        cond_2 = (np.abs(RD) / self.target_RI) <= self.RI_err
        cond_3 = (np.abs(AD) / self.target_AI) <= self.AI_err
        return (cond_1 and cond_2 and cond_3)
    
    ## Generation process
    def generate_stage_I(self):
        self.update_A2() 
        particle = self.IFFT_particle(stage=1)
        self.particle = particle
        print(f'Stage-I: {particle.calc_shape_indexes()}')
        return particle.calc_shape_indexes()

    def generate_stage_II(self, RI_1st):
        max_iters = 100
        num_iters = 0
        while True:
            num_iters += 1; print(f'Iteration_number(Stage-II): {num_iters}')
            self.update_A3(RI_1st)
            particle = self.IFFT_particle(stage=2)
            EI = particle.calc_elongation()
            RI = particle.calc_roundness()
            if self.check_stage_II(EI, RI):
                print(f'Stage-II: {particle.calc_shape_indexes()}; Iteration: {num_iters}')
                self.particle = particle
                return particle.calc_shape_indexes()
            if num_iters > max_iters:
                self.particle = particle
            #     print(f'Stage-II: {particle.calc_shape_indexes()}; Iteration: MAX_ITERS')
                return particle.calc_shape_indexes()

    def generate_stage_III_inner(self, RI_2nd):
        max_iters = 100
        num_iters = 0
        while True:
            num_iters += 1; print(f'Iteration_number(Stage-IIIa): {num_iters}')
            self.update_A16(RI_2nd)
            particle = self.IFFT_particle(stage=3)
            EI = particle.calc_elongation()
            RI = particle.calc_roundness()
            AI = particle.calc_angularity()
            if self.check_stage_III_from_RI(EI, RI, AI):
                # print(f'Stage-III-inner: {particle.calc_shape_indexes()}; Iteration: {num_iters}')
                return particle.calc_shape_indexes()
            # if num_iters > max_iters:
            #     # print(f'Stage-III-inner: {particle.calc_shape_indexes()}; Iteration: MAX_ITERS')
            #     return particle.calc_shape_indexes()

    def generate_stage_III(self, RI_2nd, AI_2nd):
        max_iters = 100
        num_iters = 0
        while True:
            num_iters += 1; print(f'Iteration_number(Stage-IIIb): {num_iters}')
            _, _, AI_3rd = self.generate_stage_III_inner(RI_2nd)
            AD = self.target_AI - AI_3rd
            A37 = np.power((np.abs(AD) / 1.4454), (1.0 / 1.6076)) / 1000.0
            RD = -44.9054*np.power(10.0 * A37, 1.9432)
            if self.check_stage_III_from_AI(RD):
                self.update_A37(A37)
                particle = self.IFFT_particle(stage=4)
                EI = particle.calc_elongation()
                RI = particle.calc_roundness()
                AI = particle.calc_angularity()
                if self.check_stage_IV(EI, RI, AI):
                    self.particle = particle
                    print(f'Stage-III: {particle.calc_shape_indexes()}; Iteration: {num_iters}')
                    return particle.calc_shape_indexes()
            # if num_iters > max_iters:
            #     self.particle = particle
            #     print(f'Stage-III: {particle.calc_shape_indexes()}; Iteration: MAX_ITERS')
            #     return particle.calc_shape_indexes()
            
    def generate_stage_IV(self, EI_3rd, RI_3rd, AI_3d):
        max_iters = 1000
        num_iters = 0
        while True:
            num_iters += 1; print(f'Iteration_number(Stage-IV): {num_iters}')
            phases = np.exp(1j*np.random.uniform(-np.pi, np.pi, 128))
            self.phases[36:64] = phases[36:64]
            self.phases[-38:-65:-1] = phases[-38:-65:-1]
            particle = self.IFFT_particle(stage=4)
            EI = particle.calc_elongation()
            RI = particle.calc_roundness()
            AI = particle.calc_angularity()
            if self.check_stage_IV(EI, RI, AI):
                self.particle = particle
                print(f'Stage-IV: {particle.calc_shape_indexes()}; Iteration: {num_iters}')
                return particle.calc_shape_indexes()
            if num_iters > max_iters:
                self.particle = particle
                print(f'Stage-IV: {particle.calc_shape_indexes()}; Iteration: MAX_ITERS')
                return particle.calc_shape_indexes()


if __name__ == '__main__':
    # User input parameters
    target_EI = 1.6986
    target_RI = 0.8267
    target_AI = 4.4617

    np.random.seed(22)
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif=['Times'])
    #plt.rcParams.update({'font.size': 14}) 

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4), sharey=False, layout='constrained')
    # VBS workflow
    np.seterr(divide='ignore')
    shape_factory = VBSGenerator(target_EI, target_RI, target_AI)
    # FIRST stage...
    _, RI_1st, _ = shape_factory.generate_stage_I()
    ax[0].set_aspect(1.0)
    ax[0].set_box_aspect(1.0)
    #ax[0].set_xlim([-1.5, 1.5])
    #ax[0].set_ylim([-1.5, 1.5])
    ax[0].set_xticks([-1, 0, 1])
    ax[0].set_yticks([-1, 0, 1])
    ax[0].set_title(r'Stage-I $(A_{2})$')
    shape_factory.particle.render(ax[0], 'lightgray', add_rect=False)
    #print(f'A2={shape_factory.A2}, A3={shape_factory.A3}, A16={shape_factory.A16}, A37={shape_factory.A37}')
    # SECOND stage...
    _, RI_2nd, AI_2nd = shape_factory.generate_stage_II(RI_1st)
    ax[1].set_aspect(1.0)
    ax[1].set_box_aspect(1.0)
    #ax[1].set_xlim([-1.5, 1.5])
    #ax[1].set_ylim([-1.5, 1.5])
    ax[0].set_xticks([-1, 0, 1])
    ax[0].set_yticks([-1, 0, 1])
    ax[1].set_title(r'Stage-II $(A_{3})$')
    shape_factory.particle.render(ax[1], 'lightgray', add_rect=False)
    #print(f'A2={shape_factory.A2}, A3={shape_factory.A3}, A16={shape_factory.A16}, A37={shape_factory.A37}')
    # THIRD stage...
    EI_3rd, RI_3rd, AI_3rd = shape_factory.generate_stage_III(RI_2nd, AI_2nd)
    ax[2].set_aspect(1.0)
    ax[2].set_box_aspect(1.0)
    #ax[2].set_xlim([-1.5, 1.5])
    #ax[2].set_ylim([-1.5, 1.5])
    ax[0].set_xticks([-1, 0, 1])
    ax[0].set_yticks([-1, 0, 1])
    ax[2].set_title(r'Stage-III $(A_{16})$')
    shape_factory.particle.render(ax[2], 'lightgray', add_rect=False)
    #print(f'A2={shape_factory.A2}, A3={shape_factory.A3}, A16={shape_factory.A16}, A37={shape_factory.A37}')
    # FOURTH stage...
    shape_factory.generate_stage_IV(EI_3rd, RI_3rd, AI_3rd)
    ax[3].set_aspect(1.0)
    ax[3].set_box_aspect(1.0)
    #ax[3].set_xlim([-1.5, 1.5])
    #ax[3].set_ylim([-1.5, 1.5])
    ax[0].set_xticks([-1, 0, 1])
    ax[0].set_yticks([-1, 0, 1])
    ax[3].set_title(r'Stage-IV $(A_{37})$')
    shape_factory.particle.render(ax[3], 'lightgray', add_rect=False)
    #print(f'A2={shape_factory.A2}, A3={shape_factory.A3}, A16={shape_factory.A16}, A37={shape_factory.A37}')
    fig.savefig('VBS.svg', transparent=True)
    plt.show()
