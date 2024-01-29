import arrayfire as af
import numpy as np
from .Mesh import *

class SolverField:
    def __init__(self, pmesh, index):
        self.profile = af.constant(0.0, pmesh.Nx, pmesh.Ny, pmesh.Nt, 1, dtype=af.Dtype.f64)
        self.pmesh = pmesh
        self.gnlse_index = index

    def save_field(self, save_dir):
        path = f"{save_dir}\\SF_{self.gnlse_index}_{self.pmesh.current_step_number}.af"
        af.save_array("SF", self.profile, path)

    def load_field(self, load_dir, step):
        path = f"{load_dir}\\SF_{self.gnlse_index}_{step}.af"
        self.profile = af.read_array(path, "SF")
        self.pmesh.current_step_number = step

    def add_field_numpy(self, numpy_array):
        self.profile += af.interop.from_ndarray(numpy_array)


    def plot(self, window):
        if self.pmesh.dims == 1:
            # Implement 1D plotting logic here
            pass
        elif self.pmesh.dims == 2:
            max_value = af.max(af.abs(self.profile))
            Z =  af.transpose((af.abs(self.profile)/max_value)).as_type(af.Dtype.f32)
            Z=Z[::-1,:]
            Z[:, 0]= 0
            Z[:,-1] = 0
            Z[0,:] = 0
            Z[-1,:] = 0
            #window.set_axes_limits(0, self.pmesh.Lx, 0, self.pmesh.Ly)
            window.set_colormap(af.COLORMAP.HEAT)
            #window.set_axes_limits(0, self.pmesh.Lx, 0, self.pmesh.Ly)
            window.image(Z)            
            window.show()
            pass
        elif self.pmesh.dims == 3:
            # Implement 3D plotting logic here (note: 3D plotting might be limited)
            pass

    def eval(self):
        self.profile.eval()

    def isempty(self):
        return self.profile.isempty()
    
    def print(self):
        print(self.profile)
    
if __name__ == "__main__":
    mesh=Mesh( Nx=10,Ny=10,Nt=1,Nz=1, dx = 0.1, dy=0.1,dt=0.1,dz=0.1)
    field = Field(mesh, index=1)
    field.print()
    