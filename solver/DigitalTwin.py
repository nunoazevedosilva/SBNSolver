import numpy as np
import arrayfire as af
from .SolverField import *
from .Mesh import *
from .aux_solver import *
from .SbnSolver import *
import matplotlib.pyplot as plt


class DigitalTwin:
    
    def __init__(self, lx=2e-3, ly=1e-3,lz=20e-3, Isat=150, V=400, alpha = 100, c1=1, c2=1):
        
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.V = V
        
        #light fluid wavelength
        lf=532*10**-9

        red_factor = 1

        #k_fluid
        kf = (2*np.pi)/(lf)

        #parameters of the crystal
        ne = 2.36 #index of refraction n_e
        r33=235*10**-12 #pm/V
        #Biasing electric Field
        E0=2*self.V*10**2 # V/m
        
        self.Isat = Isat

        #maximum index variation
        self.delta_n_max = 0.5*ne**3*r33*E0
        self.delta_n=self.delta_n_max

        #absorption
        self.alpha = alpha / (self.delta_n_max * kf) #1.319 cm^-2

        #coupling terms
        self.c1 = c1
        self.c2 = c2

        #healing length
        self.hl = 1/(kf*np.sqrt(ne*self.delta_n/self.Isat))

        #transverse direction scaling factor
        self.factor_t = kf*np.sqrt(ne*self.delta_n/red_factor)

        #propagation direction scaling factor
        self.factor_z = kf*self.delta_n

        #in adimensional units
        self.lx_sim = self.factor_t*self.lx
        self.ly_sim = self.factor_t*self.ly
        self.lz_sim = self.factor_z*self.lz

        ##################################################

        print("delta_n -> " + str(self.delta_n))
        print("lx -> " + str(self.lx_sim))
        print("ly -> " + str(self.ly_sim))
        print("lz -> " + str(self.lz_sim))
        print('healing length - > ' + str(self.hl))
        
        
        ###########################################

        
    def start_simulation_config(self, save_dir,dims=2, Nx = 256, Ny = 256, dz = 0.05, stride = 50):


        ########################################
        #######################################

        self.dims = dims
        self.Nx = Nx
        self.Ny = Ny
        
        #spatial steps
        self.dx = self.lx_sim / self.Nx
        self.dy = self.ly_sim / self.Ny
        self.dt = 1.
        print('dx->',self.dx,'dy->',self.dy)

        #integration parameters

        self.stride = stride
        self.dz = dz
        self.total_steps = int(self.lz_sim/(stride*dz))
        print('total steps to simulate ->', self.total_steps)
        self.sim_mesh=Mesh(Nx=self.Nx,Ny=self.Ny,Nt=1,Nz=1, dx = self.dx, dy=self.dy,dz=self.dz)

        ########################################
        #######################################


        self.field1 = SolverField(self.sim_mesh, index=1)
        self.field2 = SolverField(self.sim_mesh, index=2)
        self.xx,self.yy = self.sim_mesh.coords()

        ########################################
        #######################################


        self.save_dir = save_dir

        self.solver = SbnSolver(pmesh = self.sim_mesh,envelope=self.field1,envelope2=self.field2,
                           l_factor=0.5,l_factor2=0.5,
                           c1=self.c1,c2=self.c2,Isat=self.Isat,alpha=self.alpha)


    def run(self, backend = 2, device = 0):
        win = af.graphics.Window(width=600, height=600, title='Envelope1')
        win2 = af.graphics.Window(width=600, height=600, title='Envelope2')

        self.solver.simulate(stride=self.stride,total_strides=self.total_steps,
                        save_dir=self.save_dir,
                        win=win,win2=win2)

        win.close()
        win2.close()


        
    def get_E1(self):
        return af.interop.to_array(self.field1.profile)
    
    def get_E2(self):
        return af.interop.to_array(self.field2.profile)
    
    def plot(self):
        if self.dims == 2:
            field1 = np.transpose(self.get_E1())
            field2 = np.transpose(self.get_E2())
            
            fig,ax = plt.subplots(2,2,figsize=[5,5])
            
            lx = self.lx*1e3
            ly = self.ly*1e3
            
            ax[0,0].set_title(r'$|E_1|^2$')
            im = ax[0,0].imshow(np.abs(field1)**2, extent=[0,lx,0,ly],origin='lower')
            ax[0,0].set_xlabel(r'$x (mm)$')
            ax[0,0].set_ylabel(r'$y (mm)$')
            plt.colorbar(im, ax=ax[0,0])
            
            ax[0,1].set_title(r'$Phase (E_1)$')
            im1 = ax[0,1].imshow(np.angle(field1), extent=[0,lx,0,ly],cmap='seismic',vmin=-np.pi, vmax=np.pi,origin='lower')
            ax[0,1].set_xlabel(r'$x (mm)$')
            ax[0,1].set_xlabel(r'$y (mm)$')
            plt.colorbar(im1, ax=ax[0,1])
            
            ax[1,0].set_title(r'$|E_2|^2$')
            im2=ax[1,0].imshow(np.abs(field2)**2, extent=[0,lx,0,ly],origin='lower')
            ax[1,0].set_xlabel(r'$x (mm)$')
            ax[1,0].set_ylabel(r'$y (mm)$')
            plt.colorbar(im2, ax=ax[1,0])
            
            ax[1,1].set_title(r'$Phase (E_2)$')
            im3 = ax[1,1].imshow(np.angle(field2), extent=[0,lx,0,ly],cmap='seismic',vmin=-np.pi, vmax=np.pi,origin='lower')
            ax[1,1].set_xlabel(r'$x (mm)$')
            ax[1,1].set_xlabel(r'$y (mm)$')
            plt.colorbar(im3, ax=ax[1,1])
            
            
            fig.tight_layout()
        else:

            pass
        

        
    
    def initial_state_from_data(self, amplitude, phase, x_slm, y_slm, center=True):
        import scipy 

        x_slm = x_slm-np.min(x_slm)
        y_slm = y_slm-np.min(y_slm)
        #print(x_slm, y_slm)
        XX,YY = np.meshgrid(x_slm, y_slm, indexing='ij') 
        if center:
            center_of_mass = [np.sum(XX*amplitude)/np.sum(amplitude), np.sum(YY*amplitude)/np.sum(amplitude)]
            print(center_of_mass)
            roll_x = int(len(x_slm)/2-np.abs(x_slm - center_of_mass[0]).argmin())
            roll_y = int(len(y_slm)/2-np.abs(y_slm - center_of_mass[1]).argmin())
            print(roll_x, roll_y)
            amplitude = np.roll(amplitude, (roll_x,roll_y), axis=(0,1))
            phase = np.roll(phase, (roll_x,roll_y), axis=(0,1))
            
        interpolator_amplitude = scipy.interpolate.interp2d(x_slm, y_slm, np.transpose(np.abs(amplitude)), 
                    kind='linear')

        interpolator_phase = scipy.interpolate.interp2d(x_slm, y_slm, np.transpose(phase), 
                            kind='linear')
        
        return np.transpose(interpolator_amplitude(self.x_sim, self.y_sim)), np.transpose(interpolator_phase(self.x_sim, self.y_sim))
    

if __name__=='__main__':

    SBN_sim = DigitalTwin(alpha=0,c1=1,c2=1)
    SBN_sim.start_simulation_config(save_dir='test1\\',stride=25)
    xx,yy = SBN_sim.xx, SBN_sim.yy

    

    x0=SBN_sim.sim_mesh.Lx/2
    temp_field = np.exp(-(xx-x0)**2 / wx**2)
    SBN_sim.field1.add_field_numpy(temp_field)

    x1=SBN_sim.sim_mesh.Lx/3
    temp_field = np.exp(-(xx-x1)**2 / wx**2)*np.exp(1j*0.1*xx)
    SBN_sim.field2.add_field_numpy(temp_field)

    SBN_sim.plot()

    SBN_sim.run()
