# from hamiltonian_prediction import *
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from hamiltonian_prediction import *

import timeit

### questions organizer dictionary
questions = {'conj': {'Q6': [0, 1],
                      'Q8': [2, 3],
                      'Q12': [0, 3],
                      'Q16': [1, 2]},
             'disj':{'Q10': [0, 2],
                     'Q18': [1, 3]},
             'trap': {'Q14': 1},
             'Gender': 'Q3',
             'Age' : 'Q4',
             'Education' : 'Q5'}

questions_fal = {'Q10': 2,
                 'Q18': 2,
                 'Q12': 1,
                 'Q16': 1}

### which options correspond to which qubits
questions_options = {
    'Q6' : {'pa':{'0': 11},
            'pb':{'1': 16},
            'pab':{'01': 17}},
    'Q8': {'pa':{'2': 8},
           'pb':{'3': 5},
           'pab':{'23': 6}},
    'Q10' : {'pa':{'0': 4},
            'pb':{'2': 5},
            'pab':{'02': 12}},
    'Q12' : {'pa':{'0': 7},
            'pb':{'3': 4},
            'pab':{'03': 17}},
    'Q16' : {'pa':{'1': 6},
            'pb':{'2': 4},
            'pab':{'12': 8}},
    'Q18' : {'pa':{'1': 4},
            'pb':{'3': 6},
            'pab':{'13': 9}}
}


### init {probs} state in each question
prob_q = {'0' : 0, '1' : 0, '2' : 0, '3' : 0}

### init {h} state in each question
hq_q = {'0' : 0, '1' : 0, '2' : 0, '3' : 0,
        '01' : 0, '02' : 0, '03' : 0,
        '12' : 0, '13' : 0, '23' : 0,
        }

qubits_dict = {0:'a', 1:'b', 2:'c', 3:'d'}
fal_dict = {1:'C', 2: 'D'}

def q_qubits_fal(q):
    if q in list(questions['conj'].keys()):
        all_q = questions['conj'][q]
        fal = 'C'
    elif q in list(questions['disj'].keys()):
        all_q = questions['disj'][q]
        fal = 'D'
    return all_q, fal


def reformat_data_from_qualtrics(path):
    '''reformat the data from qualtrics to cleaner dataframe'''
    ### load the file
    raw_df = pd.read_csv(path)
    clms = raw_df.columns
    raw_df = raw_df.iloc[2:]

    ### clear users that fail the trap question
    ### change the range from qualtrics to [0,6]
    vd = dict(zip(np.sort(raw_df[list(questions['trap'].keys())[0]].astype('float').unique()), np.arange(6)))
    raw_df[list(questions['trap'].keys())[0]] = raw_df[list(questions['trap'].keys())[0]].astype('float').replace(vd)
    raw_df = raw_df[raw_df[list(questions['trap'].keys())[0]].astype('float') == list(questions['trap'].values())[0]]

    ### order of the questions
    ### choose the columns of the order
    rand_qs = ['Q10', 'Q12', 'Q16', 'Q18']
    rand_qs = [x + '_order' for x in rand_qs]
    order_cls = raw_df[clms[clms.str.contains('FL_')]]
    renaming_dict = dict(zip(order_cls, rand_qs))
    raw_df.rename(columns=(renaming_dict), inplace=True)

    ### remove all the order of the options inside each question
    clms = raw_df.columns
    clms = clms[~clms.str.contains('_DO_')]
    raw_df = raw_df[clms]

    # ### calculate all the aprameters and progpogate psi for the first 2 questions
    # all_data, _ = calc_first_2_questions(raw_df)

    cnames = []

    ### questions with fallacies
    fallacy_qs = list(questions['conj'].keys()) + list(questions['disj'].keys())
    id_qs = list(questions['trap'].keys()) + [questions['Gender']]+ [questions['Age']]+ [questions['Education']] + ['survey_code']
    all_cls = fallacy_qs + id_qs

    ### subsample the columns that i need
    for q in all_cls:
        cnames = cnames + list(clms[clms.str.contains(q)])

    raw_df = raw_df[cnames]
    clms = raw_df.columns

    # ### change options numbering
    # for q in fallacy_qs:
    #     cc = clms[clms.str.contains(q)]
    #     cc = cc[~cc.str.contains('order')]
    #
    #     a = cc.str.split('_', expand=True) ### current numbering
    #     list(a.levels[1]).sort()
    #     d = {}
    #     for i, j in enumerate(a.levels[1]):
    #         d[q+'_'+j] = q + '_' + str(i) ### new order
    #
    #     raw_df.rename(columns=(d), inplace=True)

    ### match option with which qubit and probability it is
    q_dict = {}
    # probs = ['pa','pb','pab']
    for q in fallacy_qs:
        for i, (p, qd) in enumerate(questions_options[q].items()):
            qubit = list(qd.keys())[0]
            option = list(qd.values())[0]
            current_name = q + '_' + str(option)
            new_name = q + '_' + 'q' + str(qubit)+ '_' + p + '_'
            q_dict[current_name] = new_name

    raw_df.rename(columns=(q_dict), inplace=True)

    raw_df = raw_df[list(q_dict.values()) + id_qs + list(raw_df.columns[raw_df.columns.str.contains('order')])]

    raw_df[list(q_dict.values())] = raw_df[list(q_dict.values())].astype('float') / 100
    # raw_df[list(q_dict.values())] = np.random.random(raw_df[list(q_dict.values())].shape)

    ### which question was third
    raw_df['q3'] = ''
    for col in raw_df.columns[raw_df.columns.str.contains('order')]:
        q = col.split('_')[0]  # questions
        raw_df.loc[raw_df[col] == '1', 'q3'] = q

    raw_df.to_csv('data/processed_data/clear_df.csv', index = False)

    return raw_df


def calc_all_questions(df):
    ### calculate all the parameters and psi for the first 2 questions

    all_data = {}
    for ui, u_id in enumerate(df['survey_code'].unique()):
        start = timeit.default_timer()

        ### init psi
        psi_0 = uniform_psi(n_qubits=4)
        sub_data = {
            'h_q': {}
        }

        ### the data of specific user
        d0 = df[(df['survey_code'] == u_id)]

        ### the order of the questions for each user
        a = d0[d0.columns[d0.columns.str.contains('order')]].reset_index(drop=True)
        a.columns = a.columns.str.replace('_order','')
        a = a + 1
        a['Q6'] = 0
        a['Q8'] = 1
        a = a.T
        a.columns = ['order']
        a = a.sort_values(by = 'order')
        qs = list(a.index)

        ### run on all questions
        for q in qs:
            p_id = a.loc[q][0]
            d = d0.copy()

            ### take the real probs of the user
            d = d[d.columns[d.columns.str.contains(q)]].reset_index(drop=True)
            p_real = {
                'A': d[d.columns[d.columns.str.contains('pa_')]].values,
                'B': d[d.columns[d.columns.str.contains('pb_')]].values,
                'A_B': d[d.columns[d.columns.str.contains('pab_')]].values
            }

            ### is the third question is conj/ disj
            all_q, fal = q_qubits_fal(q)

            sub_data[p_id] = get_question_H(psi_0, all_q, p_real, fallacy_type=fal)

            psi_0 = sub_data[p_id]['psi']

            if p_id == 0:
                sub_data[p_id]['h_q'] = hq_q.copy()
                sub_data[p_id]['prob_q'] = prob_q.copy()
            else:
                sub_data[p_id]['h_q'] = sub_data[p_id-1]['h_q'].copy()
                sub_data[p_id]['prob_q'] = sub_data[p_id-1]['prob_q'].copy()

            ### update the {h} from the most recent question.
            sub_data[p_id]['h_q'][str(all_q[0])] = sub_data[p_id]['h_a']
            sub_data[p_id]['h_q'][str(all_q[1])] = sub_data[p_id]['h_b']
            sub_data[p_id]['h_q'][str(all_q[0])+str(all_q[1])] = sub_data[p_id]['h_ab']

            ### update the {probs} from the most recent question.
            sub_data[p_id]['prob_q'][str(all_q[0])] = p_real['A']
            sub_data[p_id]['prob_q'][str(all_q[1])] = p_real['B']

        ### add the questions order of a specific user
        sub_data['qs'] = qs

        ### append current user to the dict that contains all the data
        all_data[u_id] = sub_data

        stop = timeit.default_timer()
        print('user %d/%d: ' %(ui, len(df['survey_code'].unique())), stop - start)

    ### save dict with np
    np.save('data/processed_data/all_data_dict.npy', all_data)

    print(''' 
    ================================================================================
    || Done calculating {h_i} for all questions for all users.                    || 
    || Data was saved to: data/processed_data/all_data_dict.npy                   || 
    ================================================================================''')

    return all_data

def calculate_all_data_I():
    '''calculate prediction errors an all questions'''

    ### load the dataframe containing all the data
    raw_df = pd.read_csv('data/new_code/clear_df.csv')
    raw_df.rename({'survey_code': 'userID'}, axis=1, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    ### loading all the data of all the questions
    all_data = np.load('data/new_code/all_data_dict.npy').item()

    ### creating a dataframe to save all the predictions error --> for specific question group by 'qn' --> agg('mean')
    prediction_errors = pd.DataFrame()

    ### run on all users all questions but only I !!!
    for u_id, tu in all_data.items():
        start = timeit.default_timer()

        ### running from the 2nd question
        for pos, qn in enumerate(tu['qs'][1:]):
            p_id = pos + 1
            all_q, fal = q_qubits_fal(qn)

            temp = {}
            temp['id'] = [u_id]  # suer id
            temp['qn'] = [qn]    # which question
            temp['pos'] = [p_id] # question position

            temp['q1'] = [all_q[0]] # 1st qubit
            temp['q2'] = [all_q[1]] # 2nd qubit

            q1 = 'p_' + qubits_dict[temp['q1'][0]] # 1st probability
            q2 = 'p_' + qubits_dict[temp['q2'][0]] # 2nd probability
            q12 = 'p_' + qubits_dict[temp['q1'][0]] + qubits_dict[temp['q2'][0]]

            ### psi after the previous question
            psi_0 = tu[p_id-1]['psi']

            ### all probabilities from previous questions
            temp['p_a'] = [tu[p_id-1]['prob_q']['0']]
            temp['p_b'] = [tu[p_id-1]['prob_q']['1']]
            temp['p_c'] = [tu[p_id-1]['prob_q']['2']]
            temp['p_d'] = [tu[p_id-1]['prob_q']['3']]

            ### the probs that appear in the current question, taken from previous questions
            temp['p_a_pre'] = temp[q1]
            temp['p_b_pre'] = temp[q2]

            ### real probabilities in the current question
            temp[q1 + '_current'] = [tu[p_id]['p_a'][0]]
            temp[q2 + '_current'] = [tu[p_id]['p_b'][0]]

            ### take the most updated {h} from previous question that appear in current question
            h_a = [tu[p_id-1]['h_q'][str(int(temp['q1'][0]))], None, None]
            h_b = [None, tu[p_id-1]['h_q'][str(int(temp['q2'][0]))], None]

            ### predicted probabilities with I
            temp['p_a_pred_I'] = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4).flatten()[0]]
            temp['p_b_pred_I'] = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4).flatten()[0]]

            ### calculate the error from the previous probabilities with NO U.
            temp['p_a_err_real_pre'] = [(temp[q1 + '_current'][0] - temp['p_a_pre'][0]) ** 2]
            temp['p_b_err_real_pre'] = [(temp[q2 + '_current'][0] - temp['p_b_pre'][0]) ** 2]

            temp['p_a_err_real_pre_abs'] = [np.abs(temp[q1 + '_current'][0] - temp['p_a_pre'][0])]
            temp['p_b_err_real_pre_abs'] = [np.abs(temp[q2 + '_current'][0] - temp['p_b_pre'][0])]

            ### calculate the error from the full 4 qubits state with I
            temp['p_a_err_real_I'] = [(temp[q1 + '_current'][0] - temp['p_a_pred_I'][0]) ** 2]
            temp['p_b_err_real_I'] = [(temp[q2 + '_current'][0] - temp['p_b_pred_I'][0]) ** 2]

            temp['p_a_err_real_I_abs'] = [np.abs(temp[q1 + '_current'][0] - temp['p_a_pred_I'][0])]
            temp['p_b_err_real_I_abs'] = [np.abs(temp[q2 + '_current'][0] - temp['p_b_pred_I'][0])]

            ### calculate the error from uniform
            temp['p_a_err_real_uniform'] = [(temp[q1 + '_current'][0] - .5) ** 2]
            temp['p_b_err_real_uniform'] = [(temp[q2 + '_current'][0] - .5) ** 2]

            temp['p_a_err_real_uniform_abs'] = [np.abs(temp[q1 + '_current'][0] - .5)]
            temp['p_b_err_real_uniform_abs'] = [np.abs(temp[q2 + '_current'][0] - .5)]

            prediction_errors = pd.concat([prediction_errors, pd.DataFrame(temp)], axis=0)

        stop = timeit.default_timer()

        print('user running time: ', stop - start)

    np.save('data/new_code/all_data_dict.npy', all_data)

    prediction_errors.set_index('id', inplace=True)
    prediction_errors.to_csv('data/new_code/prediction_errors.csv')  # index=False)


def plot_errors(df):
    '''Boxplot of the errors per question type.
    Also calculate statistical difference between groups.'''

    ### list of the columns of the errors
    for col in list(df.columns):
        try:
            df[col] = df[col].str.replace('[', '').str.replace(']', '').astype('float')
        except:
            pass

    print('======> errors: printing only for questions in position 2')
    df = df[df['pos'] == 2]
    print(np.sqrt(df[df.columns[df.columns.str.contains('err')]].mean()).sort_values())


    ### errors descriptive data frame
    a = df[df.columns[df.columns.str.contains('p_a')]]
    a = a[a.columns[~a.columns.str.contains('p_ab')]]

    b = df[df.columns[df.columns.str.contains('p_b')]]

    b.columns = a.columns

    a = a.append(b)
    a = a[a.columns[a.columns.str.contains('err')]]

    a.reset_index(inplace=True, drop=True)

    aa = np.sqrt(a.mean())
    print(aa.sort_values())

    print()


def main():
    # reformat, calc_questions = True , False
    reformat, calc_questions = False, True
    # reformat, calc_questions = False, False

    #calc_errs = True
    calc_errs = False

    if reformat:
        raw_df = reformat_data_from_qualtrics('data/raw_data/Emma_and_Liz_april2019_no_slider.csv')
    else:
        raw_df = pd.read_csv('data/processed_data/clear_df.csv')

    if calc_questions:
        calc_all_questions(raw_df) ### calculate all the data

    ### calcualte and predict erros of I
    ### TODO: delete before submting the paper!
    # if calc_errs:
    #     calculate_all_data_I() ### calculate error predictions
    # else:
    #     prediction_errors = pd.read_csv('data/new_code/prediction_errors.csv')
    #     plot_errors(prediction_errors)


if __name__ == '__main__':
    main()
