import hamiltonian_prediction as hp
import pandas as pd
import numpy as np
from itertools import product


def calc_p_h_irr():
    '''simulating data
    for {h} between -e to e wi ns steps
    calculating: pa, pb, pab_conj, pab_disj, irr'''
    s = 0.01 # start
    e = 1 # end
    ns = 10  # half of the number of elements

    ### for probaiblities range
    ap = np.linspace(s,e, ns)

    ### for {h} range
    ah = np.linspace(-e,e, 2 * ns + 1)

    ### create all the combinations
    b = product(ah,ah,ah)

    ### --> to np.array
    c = np.array(list(b))
    # c = np.array([[-0.75, -0.75, -0.75]])

    ### --> to pandas dataframe
    # df = pd.DataFrame(data = c, columns = ['pa','pb','ha','hb','hab'])
    df = pd.DataFrame(data = c, columns = ['ha','hb','hab'])


    ### init psi
    psi0 = hp.uniform_psi(4, 'uniform')

    ### calculate the proability of the conjunction based on all {hi} and the possible probs.
    df['pa'] = df.apply(lambda x: hp.get_general_p([x['ha'], None, None], [0,1], '0', psi0)[0][0], axis = 1)
    df['pb'] = df.apply(lambda x: hp.get_general_p([None, x['hb'], None], [0,1], '1', psi0)[0][0], axis = 1)
    df['pab_c'] = df.apply(lambda x: hp.get_general_p([None, None, x['hab']], [0,1], 'C', psi0)[0][0], axis = 1)
    df['pab_d'] = df.apply(lambda x: hp.get_general_p([None, None, x['hab']], [0,1], 'D', psi0)[0][0], axis = 1)

    ### calculate irrationalities
    df['irr_conj'] = df.apply(lambda x: x['pab_c'] - x[['pa','pb']].min(), axis = 1)
    df['irr_disj'] = df.apply(lambda x: x[['pa','pb']].max() - x['pab_d'], axis = 1)

    ### save dataframe
    df.to_csv('data/simulated_data/p_from_h.csv')

calc_p_h_irr()