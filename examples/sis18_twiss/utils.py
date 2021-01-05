import numpy as np
import pickle
import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)

from tm_pnn.layers.Taylor_Map import TaylorMap
from tm_pnn.regularization.symplectic import get_reg_term_2_2 as sympl_reg

from sis18_cell_slice import ring as sequence



def get_sequential_model():
    dim = 3
    order = 1
    model = Sequential()
    lengths = []
    for i, (R, T, name, length) in enumerate(get_transfermaps(dim=6)):
        Rx, Ry, RD, yD = twiss_transport_matrx(R)

        element_map = TaylorMap(output_dim = dim, order = order, input_shape = (dim,),
                            weights=[np.zeros((1,dim)), Rx.T],
                            weights_regularizer=lambda W: sympl_reg(0.009, W))
        element_map.tag = name
    
        model.add(element_map)
        
        lengths.append(length)
        
    lengths = np.cumsum(np.array(lengths))
        
    return model, lengths

def get_elementwise_model():
    model, lengths = get_sequential_model()
    model = Model(inputs=model.input, outputs=[el.output for el in model.layers])
    return model, lengths
    
def get_transfermaps(dim = 2):
    #sequence = get_lattice()
    method = MethodTM()
    method.global_method = SecondTM

    
    lattice = MagneticLattice(sequence,  method=method)
    for i, tm in enumerate(get_map(lattice, lattice.totalLen, Navigator(lattice))):
        R = tm.r_z_no_tilt(tm.length, 0) [:dim, :dim]
        T = tm.t_mat_z_e(tm.length, 0)[:dim, :dim, :dim].reshape((dim, -1))
        yield R, T, type(lattice.sequence[i]).__name__, lattice.sequence[i].l