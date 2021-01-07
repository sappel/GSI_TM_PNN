import numpy as np
import matplotlib.pyplot as plt

from ocelot import *
from ocelot.cpbd.optics import *
from ocelot.cpbd.moga import *
from ocelot.gui.accelerator import *

# sis18 cell as ocelot input file
from sis18_cell import ring as sequence
    
# get tansfermaps up to second ordner form the ocelot code
def get_transfermaps(dim = 2):
    method = MethodTM()
    method.global_method = SecondTM

    lattice = MagneticLattice(sequence,  method=method)

    for i, tm in enumerate(get_map(lattice, lattice.totalLen, Navigator(lattice))):
        R = tm.r_z_no_tilt(tm.length, 0)[:dim, :dim]
        T = tm.t_mat_z_e(tm.length, 0)[:dim, :dim, :dim].reshape((dim, -1))
        yield R, T, type(lattice.sequence[i]).__name__, lattice.sequence[i].l


def main():
    method = MethodTM()
    method.global_method = SecondTM

    lattice = MagneticLattice(sequence,  method=method)

    tws = twiss(lattice)
    plot_opt_func(lattice,tws, legend=False)
    plt.show()

    return


if __name__ == "__main__":
    main()