import arrayfire as af
import os
from .Mesh import *
from .SolverField import *
from .aux_solver import *

class SbnSolver:
    def __init__(self, pmesh, envelope, envelope2, l_factor, l_factor2, c1, c2, Isat,alpha):
        self.pmesh = pmesh
        self.envelope = envelope
        self.envelope2 = envelope2
        self.l_factor = l_factor
        self.l_factor2 = l_factor2
        self.c1 = complex(0, 1) *c1
        self.c2 = complex(0, 1) *c2
        self.Isat = Isat
        self.alpha = alpha

        self.k2 = k2_nd(pmesh.dx, pmesh.dy, pmesh.dt, envelope.profile.dims(), dtype=af.Dtype.f64)

    def linear_step(self, dz):
        self.envelope.profile = af.signal.fft2(self.envelope.profile)
        self.envelope.profile = self.envelope.profile*af.exp(- 1j* dz * self.k2 * self.l_factor) #/ (self.pmesh.Nx * self.pmesh.Ny * self.pmesh.Nt)
        self.envelope.profile = af.signal.ifft2(self.envelope.profile)


        self.envelope2.profile = af.signal.fft2(self.envelope2.profile)
        self.envelope2.profile = self.envelope2.profile*af.exp(- 1j* dz * self.k2 * self.l_factor) #/ (self.pmesh.Nx * self.pmesh.Ny * self.pmesh.Nt)
        self.envelope2.profile = af.signal.ifft2(self.envelope2.profile)




    def nonlinear_step(self, dz):

        envelope_abs_square = af.abs(self.envelope.profile)*af.abs(self.envelope.profile)
        envelope2_abs_square = af.abs(self.envelope2.profile)*af.abs(self.envelope2.profile)

        self.envelope.profile = self.envelope.profile * af.exp(-self.c1 * dz * ((envelope_abs_square + envelope2_abs_square) / (envelope_abs_square + envelope2_abs_square + self.Isat)))
        self.envelope2.profile = self.envelope2.profile * af.exp(-self.c2 * dz * ((envelope_abs_square + envelope2_abs_square) / (envelope_abs_square + envelope2_abs_square + self.Isat)))       
        self.envelope.profile = self.envelope.profile *np.exp(-self.alpha*dz)
        self.envelope2.profile = self.envelope2.profile *np.exp(-self.alpha*dz)



    def push(self, stride):            
            
        # First a half linear step
        self.linear_step(self.pmesh.dz / 2)

        for step in range(stride - 1):
            self.nonlinear_step(self.pmesh.dz)
            self.linear_step(self.pmesh.dz)

        # Final coupled step and half linear step to end the stride
        self.nonlinear_step(self.pmesh.dz)
        self.linear_step(self.pmesh.dz / 2)

        # Update step number and time coordinate
        self.pmesh.current_step_number += stride
        self.pmesh.current_z += self.pmesh.dz * stride



    def simulate(self, stride, total_strides, save_dir, win, win2):

        save_dir_gnlse = os.path.join(save_dir, "gnlse_field")
        os.makedirs(save_dir_gnlse, exist_ok=True)

        save_dir_gnlse2 = os.path.join(save_dir, "gnlse_field2")
        os.makedirs(save_dir_gnlse2, exist_ok=True)

        # Save mesh properties
        self.pmesh.save_parameters(save_dir)
        # Save first state
        self.envelope.save_field(save_dir_gnlse)

        # Plot first state
        self.envelope2.plot(win2)
        self.envelope.plot(win)

        for s in range(total_strides):
            print(f"Stride {s} of {total_strides}")

            # Integration step for one stride
            self.push(stride)

            # Save field at each stride
            self.envelope.save_field(save_dir_gnlse)
            self.envelope2.save_field(save_dir_gnlse2)

            # Plot envelope
            self.envelope2.plot(win2)
            self.envelope.plot(win)
        
        win.close()
        win2.close()




if __name__ == "__main__":
    
    mesh=Mesh( Nx=256,Ny=256,Nt=1,Nz=1, dx = 0.1, dy=0.1,dt=0.1,dz=0.01)
    field1 = SolverField(mesh, index=1)
    field2 = SolverField(mesh, index=2)
    xx,yy = mesh.coords()

    x0=mesh.Lx/2
    temp_field = np.exp(-(xx-x0)**2)
    field1.add_field_numpy(temp_field)

    x1=mesh.Lx/3
    temp_field = np.exp(-(xx-x1)**2)*np.exp(1j*0.1*xx)
    field2.add_field_numpy(temp_field)


    win = af.graphics.Window(width=600, height=600, title='Envelope1')
    win2 = af.graphics.Window(width=600, height=600, title='Envelope2')

    solver = SbnSolver(pmesh = mesh,envelope=field1,envelope2=field2,l_factor=0.5,l_factor2=0.5,c1=1,c2=1,Isat=100,alpha=0.0)
    solver.simulate(stride=25,total_strides=2,save_dir='test1\\',
                    win=win
                    ,win2=win2)

    win.close()
    win2.close()