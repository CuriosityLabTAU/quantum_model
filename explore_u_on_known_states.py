import numpy as np
import pandas as pd
from hamiltonian_prediction import *

hsc = ['h_a', 'h_b', 'h_c', 'h_d', 'h_ab', 'h_ac', 'h_ad', 'h_bc', 'h_bd', 'h_cd']
questions_fal = {'Q10': 'D', 'Q18': 'D', 'Q12': 'C', 'Q16': 'C'}


def get_specific_prob(O, psi, all_q, which_prob, n_qubits = 4):
    '''
    Get an operator, O, and a quantum state, psi, and return probability: p_i
    :param O: operator.
    :param psi: quantum state.
    :param all_q: qubits, e.g. [0,1] for the 1st and 2nd qubits.
    :param which_prob: pa = 0, pa = 1, pa = 'C' - conjunction or 'D' - disjunction.
    :param n_qubits: how many qubits in the quantum state.
    :return: probability, and psi after operator
    '''
    psi_dyn = np.dot(O, psi)
    P_ = MultiProjection(str(which_prob), all_q, n_qubits)
    psi_final = np.dot(P_, psi_dyn)
    p_ = norm_psi(psi_final)
    return p_.flat[0], psi_dyn

def get_probs(O, psi, all_q, which_prob, n_qubits = 4):
    '''
    Get an operator, O, and a quantum state, psi, and return probabilities: p_a, p_b, p_ab
    for more details see: get_specific_prob()'''
    probs = []
    for prob in [0,1, which_prob]:
        p, psi_dyn = get_specific_prob(O, psi, all_q, prob, n_qubits)
        probs.append(p)

    return probs, psi_dyn


### a dictionary that contains all the psies: neutral/ rational/ irrational
psies = {}

### Load dataframe of probs per {h}.
### Also load the file that creates this data frame.
df_h = pd.read_csv('data/simulated_data/p_from_h.csv')
psies['neutral'] = uniform_psi(4, 'uniform')

### take {h} most (ir)rational
### todo: (determined ONLY by h_ab --> how to choose h_a, h_b?)
h_ = {}

for r in ['rat','irr']:
    if r == 'rat':
        df_h_r = df_h[df_h['irr_conj'] == df_h['irr_conj'].max()] # most irrational h
    elif r == 'irr':
        df_h_r = df_h[df_h['irr_conj'] == df_h['irr_conj'].min()] # most rational h

    ### todo: how to take the 3 {h} and create U?
    ### sample one line from the most (ir)rational {h}
    h_a, h_b, h_ab = df_h_r.sample(1)[['ha','hb','hab']].values.flatten()
    ### take them as full_h = [h_a, h_b, h_ab] -->
    full_h = [h_a, h_b, h_ab]
    total_H = compose_H(full_h, [0,1], n_qubits=4)

    ### propogate psi_neutral with this {hs}
    psies[r] = get_psi(total_H, psies['neutral'])



    # U_rat = None
    # U_irr = None
    #
    # ### propogate psi_neutral with this {hs}
    # ### --> psi_(ir)rational
    # psies['rat'] = np.dot(U_rat, psies['neutral'])
    # psies['irr'] = np.dot(U_irr, psies['neutral'])

### load df_u90 for U
sig_h = pd.read_csv('data/predictions/sig_h_per_qn.csv')

operators = {}
for qn in sig_h['qn'].unique():
    operators['U_'+qn] = U_from_H(grandH_from_x(sig_h.loc[sig_h['qn'] == qn, hsc].values.flatten(), questions_fal[qn]))

operators['I'] = np.eye(16)

### apply U_mean_siginificant  and I on the 3 psies.
### see what happens to the states.
### from this try infer what is the meaning of U (and life)

probs_df = pd.DataFrame(columns=['operator', 'psi', 'pa', 'pb', 'pab'])

psies_final = {}
i = 0
for K, O in operators.items():
    for k, psi in psies.items():
        probs, psies_final[K+'_'+k] = get_probs(O, psi, [0,1], 'C')

        probs_df.loc[i, ['operator', 'psi', 'pa', 'pb', 'pab']] = [K, k] + probs
        i+=1

probs_df['irr_conj'] = probs_df.apply(lambda x: x['pab'] - x[['pa','pb']].min(), axis = 1)

### for python > 3.5
psies = {**psies, **psies_final}

#
# np.save('data/simulated_data/propgated_psies.npy', psies)
probs_df.to_csv('data/simulated_data/probs_vs_u_i_psies.csv')
### to load again:
# psies =  np.load('data/processed_data/propgated_psies.npy').item()

print('Finished calculating psies & probs based on U/I & rationality')
