import numpy as np
import pandas as pd
from hamiltonian_prediction import *

hsc = ['h_a', 'h_b', 'h_c', 'h_d', 'h_ab', 'h_ac', 'h_ad', 'h_bc', 'h_bd', 'h_cd']
questions_fal = {'Q10': 'D', 'Q18': 'D', 'Q12': 'C', 'Q16': 'C'}


### Load dataframe of probs per {h}.
### Also load the file that creates this data frame.
df_h = pd.read_csv('data/simulated_data/p_from_h.csv')
psi_neutral = uniform_psi(4, 'uniform')

### take {h} most (ir)rational
rat_h = df_h[df_h['irr_conj'] == 1] # most rational h
irr_h = df_h[df_h['irr_conj'] == -1] # most irrational h


### propogate psi_neutral with this {hs}
### --> psi_(ir)rational

### load df_u90 for U
sig_h = pd.read_csv('data/predictions/sig_h_per_qn.csv')
for qn in sig_h.index.unique():
    U = U_from_H(grandH_from_x(sig_h.loc[qn, hsc].values, questions_fal[qn]))



### apply U_mean_siginificant  and I on the 3 psies.
### see what happens to the states.
### from this try infer what is the meaning of U (and life)