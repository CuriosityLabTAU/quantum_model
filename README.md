# Irrational Quantum Model
This folder contains the most updated model.

On Torr's pc, the latest model is in Quantum predictions new questionnaire, but can't sync with github.

### How to work with the code and files in this folder: 

#### Reformat the raw data from Qualtrics
Taking the raw data from Qualtrics, reformat it and analyze the quantum data in the first 2 questions.
Raw data: questionnaire that people answered questions by inserting the probability from 0 to 100 (not scale).
raw_dat --> reformat_data.py --> data/processed_data/clear_df.csv        # reformatted data
                                --> data/processed_data/all_data_dict.npy   # quantum data

#### Apply the quantum model on the experimental data. 
Check the model on all the data:
clear_df, all_data_dict --> calc_explore_U -->

Predictions based on 1-, kfold   : data/predictions/kfold_prediction.csv
{h_i} were saved to              : data/predictions/df_h.csv
Dictionary of test users (per qn): saved to data/test_users.npy
Supplementary files were saved to: kfold_all_data_dict.npy, kfold_UbyQ.npy

{h} per k fold per question                       : data/predictions/df_h.csv
Significant {h} from all the k fold (per question): data/predictions/sig_h_per_qn.csv

Make predictions on 10 percent of the users that were randomly chose for a test set: data/predictions/10percent_predictions.csv

Compare different models,  U/ I/ Previous probability/ mean probability (of train) : data/predictions/model_comparison.csv

#### Apply the quantum model on simulted data.
Calculate ptobabilities based on {h_i} range, from neutral quantum state 
calc_explore_U --> data/simulated_data/p_from_h.csv

##### Check how the quantum model interacts with (ir)rationality on simulated data
explore_u_on_known_states.py --> data/simulated_data/propogated_psies.npy
                                 
most_irrational_h --> U_irr
most_rational_h   --> U_rat

|psi_neutral>
|psi_rational> = U_irr |psi_neutral>
|psi_irrational> = U_rat |psi_neutral>

U_sig_h <-- U from the most significant h from the real data 

Check how te probabilities change from different psi.
U_sig_h |psi_X>
I |psi_X>

from its calculate the probabilities and irrationalities and save to --> data/simulated_data/probs_vs_u_i_psies.csv
