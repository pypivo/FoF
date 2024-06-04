import numpy as np
import json
import info_about_model as model
from coefficient_controller import CoefficientsController

import os
import torch
from local_contributor_config import problem_folder
from local_contributor_config import params

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

with open('parameters.json') as f:
    data = json.load(f)
print(data)

lambda_ = data["lambda_"]
sigma = data["sigma"]
alpha_base = data["alpha_base"]
beta_base = data["beta_base"]
gamma_base = data["gamma_base"]
CL_GLN_base = data["CL_GLN_base"]
CL_CAM_base = data["CL_CAM_base"]
CL_INS_base = data["CL_INS_base"]
t_0_input = data["t_0_input"]
tau_grid_input = data["tau_grid_input"]
INS_check_coeff = data["INS_check_coeff"]
tau_grid = data["tau_grid"]
t_0 = data["t_0"]
t_end = data["t_end"]

HeartRate = data["HeartRate"]
k_BMR_Glu_ef = data["k_BMR_Glu_ef"]
k_BMR_AA_ef = data["k_BMR_AA_ef"]
K_BMR_FFF_ef = data["K_BMR_FFF_ef"]
K_BMR_KB_ef = data["K_BMR_KB_ef"]
inv_beta_KB_ef = data["inv_beta_KB_ef"]
inv_beta_Glu_ef = data["inv_beta_Glu_ef"]
inv_beta_AA_ef = data["inv_beta_AA_ef"]
inv_beta_FFA_ef = data["inv_beta_FFA_ef"]
inv_beta_Muscle = data["inv_beta_Muscle"]
inv_beta_GG_m = data["inv_beta_GG_m"]
inv_beta_TG_a = data["inv_beta_TG_a"]
inv_beta_GG_h = data["inv_beta_GG_h"]
MASS_OF_HUMAN = data["Вес"]
E_day = data["E_day"]
e_sigma = data["e_sigma"]
beta_KB_ef = data["beta_KB_ef"]
beta_Glu_ef = data["beta_Glu_ef"]
beta_AA_ef = data["beta_AA_ef"]
beta_FFA_ef = data["beta_FFA_ef"]
beta_Muscle = data["beta_Muscle"]
beta_GG_m = data["beta_GG_m"]
beta_TG_a = data["beta_TG_a"]
beta_GG_h = data["beta_GG_h"]

velocity_depot = data["velocity_depot"]
power_of_coeff = data["power_of_coeff"]
j_base = data["j_base"]

print(j_base)

Glu_ef_start = E_day / beta_Glu_ef / 4
AA_ef_start = E_day / beta_AA_ef / 4
FFA_ef_start = E_day / beta_FFA_ef / 4
KB_ef_start = E_day / beta_KB_ef / 4

start_point_dict = {
    # Myocyte
    "Muscle_m": 50.0,
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
    "Glu_ef": Glu_ef_start,
    "AA_ef": AA_ef_start,
    "FFA_ef": FFA_ef_start,
    "KB_ef": KB_ef_start,
    "Glycerol_ef": 10.0,
    "Lac_m": 10.0,
    "TG_pl": 10.0,
    "Cholesterol_pl": 10.0,

    # Hormones
    "INS": 0.0,
    "GLN": 0.0,
    "CAM": 0.0,
}

N = int((t_end - t_0) / tau_grid) + 1
time_grid = np.linspace(start=t_0, stop=t_end, num=N)

SUBSTANCE_LIST = tuple(name for name in start_point_dict.keys())



def make_default_coefficients() -> dict:
    coefficients = {}
    for name in model.match_coefficient_name_and_input_substances.keys():
        if name in ['m_1', 'm_3', 'm_4', 'm_5']:
            coefficients[name] = 1
        elif name in model.DEPO_COEFFICIENTS:
            coefficients[name] = velocity_depot
        elif "j" in name:
            coefficients[name] = j_base
        else:
            coefficients[name] = power_of_coeff
    coefficients['m_21'] = 10 ** (-2)

    return coefficients
