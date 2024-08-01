import os

import numpy as np
import torch

from local_contributor_config import problem_folder

J_flow_prot_func = torch.load(
    os.path.join(problem_folder, 'ddt_AA_ef'))

J_flow_carb_func = torch.load(
    os.path.join(problem_folder, 'ddt_Glu_ef'))
J_flow_fat_func = torch.load(
    os.path.join(problem_folder, 'ddt_TG_pl'))
J_prot_func = torch.load(
    os.path.join(problem_folder, 'J_prot'))

J_fat_func = torch.load(
    os.path.join(problem_folder, 'J_fat'))
J_carb_func = torch.load(
    os.path.join(problem_folder, 'J_carb'))



lambda_ = 1.0
sigma = 0.07

alpha_base =        2.0
beta_base  =        0.02
gamma_base =        1.0

CL_GLN_base=1.0/10.0
CL_CAM_base=1.0/10.0
CL_INS_base=1.0/10.0

t_0_input= 0.0
tau_grid_input = 0.1
INS_check_coeff = 400 # [mmol*s]
tau_grid = 0.01 # [min]
t_0 = 400.0 # [min]

t_end = 400.0+1440.0*3 # [min]
N = int((t_end-t_0)/tau_grid)+1
time_grid = np.linspace(start=t_0, stop=t_end, num=N)

HeartRate = 80.0

k_BMR_Glu_ef = 10**(-2)
k_BMR_AA_ef = 10**(-2)
K_BMR_FFF_ef = 10**(-2)
K_BMR_KB_ef = 10**(-2)      

inv_beta_KB_ef = 1.0/(517.0/1000.0)
inv_beta_Glu_ef = 1.0/(699.0/1000.0)
inv_beta_AA_ef = 1.0/(369.5/1000.0)
inv_beta_FFA_ef = 1.0/(2415.6/1000.0)
inv_beta_Muscle = 1.0/(369.5/1000.0)
inv_beta_GG_m = 1.0/(699.0/1000.0)
inv_beta_TG_a = 1.0/(7246.8/1000.0)
inv_beta_GG_h = 1.0/(699.0/1000.0)

MASS_OF_HUMAN = 70.0
E_day = 1500.0 # [kcal/day]
e_sigma = E_day/(24.0*60.0) #[kcal/min]

beta_Glu_ef = 757.0/1000.0 # [kcal/mmol]
beta_AA_ef = 462/1000.0 # [kcal/mmol]
beta_KB_ef = 437/1000.0 # [kcal/mmol]
beta_TG_pl = 8 # [kcal/mmol]

beta_FFA_ef = 2415.6/1000.0 # [kcal/mmol]
beta_Muscle = 369.5/1000.0  # [kcal/mmol]
beta_GG_m = 699.0/1000.0 # [kcal/mmol]
beta_TG_a = 7246.8/1000.0 # [kcal/mmol]
beta_GG_h = 699.0/1000.0 # [kcal/mmol]


base_BMR_Glu_ef_rate = 2 * 10**(-2)  # 20% из 0.1 [kcal/0.1min (6 sec)]
base_BMR_AA_ef_rate = 10**(-2)
base_BMR_KB_ef_rate = 0.35 * 10**(-2) 

base_BMR_Glu_ef = base_BMR_Glu_ef_rate * 1/beta_Glu_ef
base_BMR_AA_ef = base_BMR_AA_ef_rate * 1/beta_AA_ef
base_BMR_KB_ef = base_BMR_KB_ef_rate * 1/beta_KB_ef

Glu_ef_start= E_day/beta_Glu_ef/4
AA_ef_start = E_day/beta_AA_ef/4
FFA_ef_start = E_day/beta_FFA_ef/4
KB_ef_start = E_day/beta_KB_ef/4

start_point_dict = {
    # Myocyte
    "Muscle_m": 10.0,
    "AA_m": 10.0,
    "GG_m": 10.0,
    "G6_m": 10.0,
    "G3_m": 10.0,
    "Pyr_m": 10.0,
    "Cit_m": 10.0,
    "OAA_m": 10.0,
    "CO2_m": 10.0,
    "H2O_m": 10.0,
    "H_cyt_m": 10.0,
    "H_mit_m": 10.0,
    "Ac_CoA_m": 10.0,
    "FA_CoA_m": 10.0,
    "ATP_cyt_m": 10.0,
    "ATP_mit_m": 10.0,

    # Adipocyte
    "TG_a": 10.0,
    "AA_a": 10.0,
    "G6_a": 10.0,
    "G3_a": 10.0,
    "Pyr_a": 10.0,
    "Ac_CoA_a": 10.0,
    "FA_CoA_a": 10.0,
    "Cit_a": 10.0,
    "OAA_a": 10.0,
    "NADPH_a": 10.0,

    # Hepatocyte
    "GG_h": 10.0,
    "G6_h": 10.0,
    "G3_h": 10.0,
    "TG_h": 10.0,
    "Pyr_h": 10.0,
    "MVA_h": 10.0,
    "OAA_h": 10.0,
    "Cit_h": 10.0,
    "AA_h": 10.0,
    "NADPH_h": 10.0,
    "Ac_CoA_h": 10.0,
    "FA_CoA_h": 10.0,

    # Fluid
    "Urea_ef": 10.0,
    "Glu_ef": 5,
    "AA_ef": 5,
    "FFA_ef": 5, 
    "KB_ef": 5,
    "Glycerol_ef": 10.0,
    "Lac_m": 10.0,
    "TG_pl": 10.0,
    "Cholesterol_pl": 10.0,    

    # Hormones
    "INS": 0.0,
    "GLN": 0.0,
    "CAM": 0.0,
}

default_coefficients = {
    # Myocyte
    "m_1": None,
    "m_2": None,
    "m_3": None,
    "m_4": None,
    "m_5": None,
    "m_6": None,
    "m_7": None,
    "m_8": None,
    "m_9": None,
    "m_10": None,
    "m_11": None,
    "m_12": None,
    "m_13": None,
    "m_14": None,
    "m_15": None,
    "m_16": None,
    "m_17": None,
    "m_18": None,
    "m_19": None,
    "m_20": None,
    "m_21": None,

    # Adipocyte
    "a_1": None,
    "a_2": None,
    "a_3": 10**(-7),
    "a_4": None,
    "a_5": None,
    "a_6": None,
    "a_7": None,
    "a_8": None,
    "a_9": None,
    "a_10": None,
    "a_11": None,
    "a_12": None,
    "a_13": None,
    "a_14": None,
    "a_15": None,
    "a_16": None,
    "a_17": None,
    "a_18": None,
    "a_19": None,

    # Hepatocyte
    "h_1": None,
    "h_2": 10**(-1),
    "h_3": None,
    "h_4": None,
    "h_5": None,
    "h_6": None,
    "h_7": None,
    "h_8": None,
    "h_9": None,
    "h_10": None,
    "h_11": None,
    "h_12": None,
    "h_13": None,
    "h_14": None,
    "h_15": None,
    "h_16": None,
    "h_17": None,
    "h_18": None,
    "h_19": None,
    "h_20": None,
    "h_21": None,
    "h_22": None,
    "h_23": None,
    "h_24": None,
    "h_25": None,
    "h_26": None,
    "h_27": None,
    "h_28": None,
    "h_29": None,

    # BMR
    "j_0": None,
    "j_1": 2  * 10**(-1),
    "j_2": None,
    "j_3": None,
    "j_4": None,

}

value_of_coeff = 1.0
j_base = 5 * 10**(-1)

def make_default_coefficients() -> dict:

    coefficients = {}
    for name in default_coefficients:
        if default_coefficients[name] is not None:
            coefficients[name] = default_coefficients[name]
            continue

        if "j" in name:
            coefficients[name] = j_base
        else:
            coefficients[name] = value_of_coeff

    return coefficients 
