import arrayfire as af
import numpy as np


class Mesh:
    def __init__(self, Nx, Ny=1, Nt=1, Nz=1, dx=1, dy=1, dt=1, dz=1):
        self.dims = len([s for s in [Nx,Ny,Nz,Nt] if s != 1])
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nt = Nt
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.Lx = Nx * dx
        self.Ly = Ny * dy
        self.Lz = Nz * dz
        self.Lt = Nt * dt
        self.current_z = 0
        self.current_step_number = 0

    def x(self):
        return self.dx * af.range(self.Nx, self.Ny, self.Nt, 1, dim=0)

    def y(self):
        return self.dy * af.range(self.Nx, self.Ny, self.Nt, 1, dim=1)

    def z(self):
        return self.dz * af.range(self.Nx, self.Ny, self.Nt, self.Nz, dim=3)

    def t(self):
        return self.dt * af.range(self.Nx, self.Ny, self.Nt, 1, dim=2)
    
    def coords(self):

        self.XX = np.array(self.x(),order='F')
        self.YY = np.array(self.y(),order='F')

        return self.XX, self.YY

    def save_parameters(self, save_dir):
        with open(f"{save_dir}/parameters.dat", "w") as param:
            param.write(f"Dimension\n{self.dims}\n")
            param.write(f"dh\n{self.dx} \n{self.dy} \n{self.dt} \n{self.dz} \n")
            param.write(f"Npoints\n{self.Nx} \n{self.Ny} \n{self.Nt} \n{self.Nz} \n")
            param.write(f"Limits\n{self.Lx} \n{self.Ly} \n{self.Lt} \n{self.Lz} \n")

    def __str__(self):
        output = (
            f"Dimension\n {self.dims}\n"
            f"dx\t Nx\n {self.dx}\t {self.Nx}\n"
            f"dy\t Ny\n {self.dy}\t {self.Ny}\n"
            f"dt\t Nt\n {self.dt}\t {self.Nt}\n"
            f"dz\t Nz\n {self.dz}\t {self.Nz}\n"
            f"Limits\nLx\t {self.Lx}\nLy\t {self.Ly}\nLt\t {self.Lt}\nLz\t {self.Lz}"
        )
        return output

    def print_xyzt(self):
        print("----- X Mesh Vector -----")
        print(self.x())
        print("----- Y Mesh Vector -----")
        print(self.y())
        print("----- Z Mesh Vector -----")
        print(self.z())
        print("----- T Mesh Vector -----")
        print(self.t())

if __name__ == "__main__":
    mesh=Mesh(Nx=10,Ny=10,Nt=1,Nz=1, dx = 0.1, dy=0.1,dt=0.1,dz=0.1)
    print(mesh)
    mesh.print_xyzt()