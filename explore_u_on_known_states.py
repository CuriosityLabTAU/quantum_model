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


def get_ps(hs, all_q, which_prob, psi_0, n_qubits=4):
    '''
    return probabilities: p_a, p_b, p_ab
    for more details see: get_specific_p()'''
    probs = []
    for prob in [0,1, which_prob]:
        if prob == 0:
            full_h = [hs[0], None, None]
        elif prob == 1:
            full_h = [None, hs[1], None]
        elif prob == which_prob:
            full_h = [None, None, hs[2]]

        p = get_specific_p(full_h, all_q, which_prob, psi_0, n_qubits)
        probs.append(p)
    return probs

def get_specific_p(full_h, all_q, which_prob, psi_0, n_qubits=4):
    '''
    Calculate the probabilit
    :param full_h: [ha, hb, hab]  e.g: [x, None, None] for calculating pa.
    :param all_q: which qubits e.g: [0,1] for 1st and 2nd qubits.
    :param which_prob: pa = 0, pa = 1, pa = 'C' - conjunction or 'D' - disjunction.
    :param psi_0: quantum state.
    :param n_qubits: how many qubits in the quantum state.
    :return:
    '''
    ### compose hamiltonian.
    H_ = compose_H(full_h, all_q, n_qubits, which_prob)

    ### propogate psi with the hamiltonian.
    psi_dyn = get_psi(H_, psi_0)

    ### projector to subspace according to which_prob that interests us.
    P_ = MultiProjection(which_prob, all_q, n_qubits)

    ### project on subspace.
    psi_final = np.dot(P_, psi_dyn)

    ### calculate the probability = norm of the state.
    p_ = norm_psi(psi_final)
    return p_.flat[0]

### a dictionary that contains all the psies: neutral/ rational/ irrational
psies = {}

### Load dataframe of probs per {h}.
### Also load the file that creates this data frame.
df_h = pd.read_csv('data/simulated_data/p_from_h.csv')
psies['neutral'] = uniform_psi(4, 'uniform')

### a dictionary that contains all the {h} per rationality.
full_h = {}
### take {h} most (ir)rational
for r in ['rat','irr']:
    if r == 'irr':
        df_h_r = df_h[df_h['irr_conj'] == df_h['irr_conj'].max()] # most irrational h
    elif r == 'rat':
        df_h_r = df_h[df_h['irr_conj'] == df_h['irr_conj'].min()] # most rational h

    ### todo: how to take the 3 {h} and create U?
    ### todo: (determined ONLY by h_ab --> how to choose h_a, h_b?)
    ### sample one line from the most (ir)rational {h}
    h_a, h_b, h_ab = df_h_r.sample(1)[['ha','hb','hab']].values.flatten()

    ### take them as full_h = [h_a, h_b, h_ab] -->
    full_h[r] = [h_a, h_b, h_ab]
    total_H = compose_H(full_h[r], [0,1], n_qubits=4)

    ### propogate psi with the hamiltonian, which composed from {h} per rationality.
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

probs_df = pd.DataFrame(columns=['operator', 'psi', 'h_type', 'pa', 'pb', 'pab'])

psies_final = {}
i = 0
for K, O in operators.items():
    for k, psi in psies.items():
        # probs, psies_final[K+'_'+k] = get_probs(O, psi, [0,1], 'C')

        psies_final[K + '_' + k] = np.dot(O, psi)
        ### Calculate the probabilities with rational & irrational {h}
        rat_net_probs = get_ps(full_h['rat'], [0,1], 'C', psies_final[K+'_'+k])
        irr_net_probs = get_ps(full_h['irr'], [0,1], 'C', psies_final[K+'_'+k])

        probs_df.loc[i,   ['operator', 'psi', 'h_type', 'pa', 'pb', 'pab']] = [K, k, 'rat'] + rat_net_probs
        probs_df.loc[i+1, ['operator', 'psi', 'h_type', 'pa', 'pb', 'pab']] = [K, k, 'irr'] + irr_net_probs
        i += 2

### caqlculate conjunction irrationality
probs_df['irr_conj'] = probs_df.apply(lambda x: x['pab'] - x[['pa','pb']].min(), axis = 1)
probs_df['is_irr'] = probs_df['irr_conj'] > 0

### for python > 3.5
psies = {**psies, **psies_final}

### for aesthetic purposes
probs_df[['pa', 'pb', 'pab','irr_conj']] = probs_df[['pa', 'pb', 'pab','irr_conj']].astype(float).round(2)

### save psies and predicted probabilities for different states
np.save('data/simulated_data/propgated_psies.npy', psies)
probs_df.to_csv('data/simulated_data/probs_vs_u_i_psies.csv', index=False)

### to load again:
# psies =  np.load('data/processed_data/propgated_psies.npy').item()
# probs_df = pd.read_csv('data/simulated_data/probs_vs_u_i_psies.csv')

print('Finished calculating psies & probs based on U/I & rationality')
