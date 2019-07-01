import numpy as np
import pandas as pd
from hamiltonian_prediction import *

hsc = ['h_a', 'h_b', 'h_c', 'h_d', 'h_ab', 'h_ac', 'h_ad', 'h_bc', 'h_bd', 'h_cd']
questions_fal = {'Q10': 'D', 'Q18': 'D', 'Q12': 'C', 'Q16': 'C'}

### a dictionary that contains all the psies
psies = {}


### Load dataframe of probs per {h}.
### Also load the file that creates this data frame.
df_h = pd.read_csv('data/simulated_data/p_from_h.csv')
psies['neutral'] = uniform_psi(4, 'uniform')

### take {h} most (ir)rational
### todo: (determined ONLY by h_ab --> how to choose h_a, h_b?)
rat_h = df_h[df_h['irr_conj'] == df_h['irr_conj'].max()] # most rational h
irr_h = df_h[df_h['irr_conj'] == df_h['irr_conj'].min()] # most irrational h

### todo: how to take the 3 {h} and create U?
### take them as full_h = [h_a, h_b, h_ab] -->
# h_a, h_b, h_ab = rat_h['ha'], rat_h['ha'], rat_h['hab']
# h_a, h_b, h_ab = irr_h['ha'], irr_h['ha'], irr_h['hab']
# full_h = [h_a, h_b, h_ab]
# total_H = compose_H(full_h, all_q, n_qubits=4)
# psi_final = get_psi(total_H, psi_0)

U_rat = None
U_irr = None

### propogate psi_neutral with this {hs}
### --> psi_(ir)rational
psies['rat'] = np.dot(U_rat, psies['neutral'])
psies['irr'] = np.dot(U_irr, psies['neutral'])

### load df_u90 for U
sig_h = pd.read_csv('data/predictions/sig_h_per_qn.csv')

for qn in sig_h.index.unique():
    U = U_from_H(grandH_from_x(sig_h.loc[qn, hsc].values, questions_fal[qn]))

    ### apply U_mean_siginificant  and I on the 3 psies.
    ### see what happens to the states.
    ### from this try infer what is the meaning of U (and life)

    psies[qn + '_net'] = np.dot(U, psies['neutral'])
    psies[qn + '_rat'] = np.dot(U, psies['rat'])
    psies[qn + '_irr'] = np.dot(U, psies['irr'])


np.save('data/predictions/propgated_psies.npy', psies)
### to load again:
# psies =  np.load('data/processed_data/propgated_psies.npy').item()