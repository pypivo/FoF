"""
информация о модели для её корректной работы
"""


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
    "Glu_ef": 0,
    "AA_ef": 0.0,
    "FFA_ef": 0, 
    "KB_ef": 0,
    "Glycerol_ef": 10.0,
    "Lac_m": 10.0,
    "TG_pl": 10.0,
    "Cholesterol_pl": 10.0,    

    # Hormones
    "INS": 0.0,
    "GLN": 10.0,
    "CAM": 10.0,
}
INS_i = 47
GLN_i = 48
CAM_i = 49


DEPO_COEFFICIENTS = [
    "a_3",
    "a_17",
    "a_18",
    "a_19",
    "a_5",
    "a_6",
    "a_8",
    "h_2",
    "h_10",
    "h_12",
    "h_14",
    "h_13",
    "h_15",
    "h_9",
    "m_7",
    "m_8",
    "m_9",
    "m_10",
    "j_0"
]

INSULIN_COEFFICIENTS = (
    'a_4', 'a_5', 'a_7', 'a_10', 'a_12', 'a_13', 'a_14',
    'h_3', 'h_7', 'h_10', 'h_12', 'h_16', 'h_17', 'h_19', 'h_20', 'h_24', 'h_26',
    'm_1', 'm_7', 'm_9', 'm_11', # 'j_0', 'a_2',
)

GLUCAGON_COEFFICIENTS = (
    'a_3', 'a_9', 'a_11',
    'h_2', 'h_6', 'h_11', 'h_13', 'h_18', 'h_23', 'h_25',
    'm_8', 
)

myocyte_coefficients_names = ( "",
    "m_1", "m_2", "m_3", "m_4", "m_5", "m_6", "m_7", "m_8", "m_9", "m_10", 
    "m_11", "m_12", "m_13", "m_14", "m_15", "m_16", "m_17", "m_18", "m_19", "m_20", "m_21",
)


adipocyte_coefficients_names = ( "",
    "a_1","a_2","a_3","a_4","a_5","a_6","a_7","a_8","a_9","a_10",
    "a_11","a_12","a_13","a_14","a_15","a_16","a_17","a_18","a_19",
)

hepatocyte_coefficients_names = ("",
    "h_1","h_2","h_3","h_4","h_5","h_6","h_7","h_8","h_9","h_10",
    "h_11","h_12","h_13","h_14","h_15","h_16","h_17","h_18","h_19","h_20",
    "h_21","h_22","h_23","h_24","h_25","h_26","h_27","h_28","h_29",
)


fluid_coefficients_names = ("",
    "j_0","j_1","j_2","j_3","j_4",
)


# входящие вещества для процесса
match_coefficient_name_and_input_substances = {
    # Myocyte
    "m_1": ['Glu_ef'],
    "m_2": ['Pyr_m', 'H_cyt_m'],
    "m_3": ['KB_ef'],
    "m_4": ['FFA_ef'],
    "m_5": ['AA_ef'],
    "m_6": ['AA_m'],
    "m_7": ['G6_m'],
    "m_8": ['GG_m'],
    "m_9": ['G6_m'],
    "m_10": ['G3_m'],
    "m_11": ['Pyr_m'],
    "m_12": ['FA_CoA_m'],
    "m_13": ['Ac_CoA_m', 'OAA_m'],
    "m_14": ['Cit_m'],
    "m_15": ['H_cyt_m'],
    "m_16": ['H_mit_m'],
    "m_17": ['AA_m'],
    "m_18": ['AA_m'],
    "m_19": ['AA_m'],
    "m_20": ['AA_m'],
    "m_21": ['Muscle_m'],
    # Adipocyte
    "a_1": ['AA_ef'],
    "a_2": ['FFA_ef'],
    "a_3": ['TG_a'],
    "a_4": ['Glu_ef'],
    "a_5": ['G6_a'],
    "a_6": ['G6_a'],
    "a_7": ['G3_a', 'FA_CoA_a'],
    "a_8": ['G3_a'],
    "a_9": ['OAA_a'],
    "a_10": ['Pyr_a'],
    "a_11": ['Pyr_a'],
    "a_12": ['OAA_a'],
    "a_13": ['Ac_CoA_a', 'NADPH_a'],
    "a_14": ['Cit_a'],
    "a_15": ['Cit_a'],
    "a_16": ['OAA_a', 'Ac_CoA_a'],
    "a_17": ['AA_a'],
    "a_18": ['AA_a'],
    "a_19": ['AA_a'],
    # Hepatocyte
    "h_1": ['AA_ef'],
    "h_2": ['G6_h'],
    "h_3": ['Glu_ef'],
    "h_4": ['Glycerol_ef'],
    "h_5": ['Lac_m'],
    "h_6": ['Ac_CoA_h'],
    "h_7": ['MVA_h'],
    "h_8": ['FFA_ef'],
    "h_9": ['TG_h'],
    "h_10": ['G6_h'],
    "h_11": ['GG_h'],
    "h_12": ['G6_h'],
    "h_13": ['G3_h'],
    "h_14": ['G6_h'],
    "h_15": ['G3_h'],
    "h_16": ['Pyr_h'],
    "h_17": ['Ac_CoA_h', 'NADPH_h'],
    "h_18": ['FA_CoA_h'],
    "h_19": ['Ac_CoA_h', 'NADPH_h'],
    "h_20": ['G3_h', 'FA_CoA_h'],
    "h_21": ['OAA_h'],
    "h_22": ['Cit_h'],
    "h_23": ['OAA_h'],
    "h_24": ['OAA_h'],
    "h_25": ['Pyr_h'],
    "h_26": ['Cit_h'],
    "h_27": ['AA_h'],
    "h_28": ['AA_h'],
    "h_29": ['AA_h'],

    "j_0": ['TG_pl'],
    "j_1": ['Glu_ef'],
    "j_2": ['KB_ef'],
    "j_3": ['FFA_ef'],
    "j_4": ['AA_ef'],

}


# выходящие вещества для процесса
match_coefficient_name_and_output_substances = {
    # Myocyte
    "m_1": ['G6_m'],
    "m_2": ['Lac_m'],
    "m_3": ['H_mit_m', 'Ac_CoA_m'],
    "m_4": ['FA_CoA_m'],
    "m_5": ['AA_m'],
    "m_6": ['AA_ef'],
    "m_7": ['GG_m'],
    "m_8": ['G6_m'],
    "m_9": ['G3_m'],
    "m_10": ['H_cyt_m', 'Pyr_m', 'ATP_cyt_m'],
    "m_11": ['Ac_CoA_m', 'CO2_m'],
    "m_12": ['Ac_CoA_m', 'H_mit_m'],
    "m_13": ['Cit_m'],
    "m_14": ['OAA_m', 'CO2_m'],
    "m_15": ['H_mit_m'],
    "m_16": ['ATP_mit_m', 'H2O_m'],
    "m_17": ['Pyr_m', 'Urea_ef'],
    "m_18": ['Ac_CoA_m', 'Urea_ef'],
    "m_19": ['OAA_m', 'Urea_ef'],
    "m_20": ['Muscle_m'],
    "m_21": ['AA_m'],
    # Adipocyte
    "a_1": ['AA_a'],
    "a_2": ['FA_CoA_a'],
    "a_3": ['Glycerol_ef', 'FFA_ef'],
    "a_4": ['G6_a'],
    "a_5": ['G3_a'],
    "a_6": ['G3_a', 'NADPH_a'],
    "a_7": ['TG_a'],
    "a_8": ['G3_a'],
    "a_9": ['OAA_a'],
    "a_10": ['Ac_CoA_a'],
    "a_11": ['OAA_a'],
    "a_12": ['NADPH_a', 'Pyr_a'],
    "a_13": ['FA_CoA_a'],
    "a_14": ['OAA_a', 'Ac_CoA_a'],
    "a_15": ['OAA_a'],
    "a_16": ['Cit_a'],
    "a_17": ['OAA_a', 'Urea_ef'],
    "a_18": ['Ac_CoA_a', 'Urea_ef'],
    "a_19": ['Pyr_a', 'Urea_ef'],
    # Hepatocyte
    "h_1": ['AA_h'],
    "h_2": ['Glu_ef'],
    "h_3": ['G6_h'],
    "h_4": ['G3_h'],
    "h_5": ['Pyr_h'],
    "h_6": ['KB_ef'],
    "h_7": ['Cholesterol_pl'],
    "h_8": ['FA_CoA_h'],
    "h_9": ['TG_pl'],
    "h_10": ['GG_h'],
    "h_11": ['G6_h'],
    "h_12": ['G3_h'],
    "h_13": ['G6_h'],
    "h_14": ['G3_h', 'NADPH_h'],
    "h_15": ['Pyr_h'],
    "h_16": ['Ac_CoA_h'],
    "h_17": ['MVA_h'],
    "h_18": ['Ac_CoA_h'],
    "h_19": ['FA_CoA_h'],
    "h_20": ['TG_h'],
    "h_21": ['Cit_h'],
    "h_22": ['OAA_h'],
    "h_23": ['G3_h'],
    "h_24": ['Pyr_h', 'NADPH_h'],
    "h_25": ['OAA_h'],
    "h_26": ['Ac_CoA_h', 'OAA_h'],
    "h_27": ['Ac_CoA_h', 'Urea_ef'],
    "h_28": ['OAA_h', 'Urea_ef'],
    "h_29": ['Pyr_h', 'Urea_ef'],

    "j_0": ['Glycerol_ef', 'FFA_ef'],
    "j_1": [],
    "j_2": [],
    "j_3": [],
    "j_4": [],

}

# показывает, в каких процессах участвуют вещества
match_substances_and_reactions = {
    'TG_a': ['a_7', 'a_3'],
    'AA_a': ['a_1', 'a_17', 'a_18', 'a_19'],
    'G6_a': ['a_4', 'a_5', 'a_6'],
    'G3_a': ['a_5', 'a_6', 'a_9', 'a_7', 'a_8'],
    'GG_h': ['h_10', 'h_11'],
    'G6_h': ['h_3', 'h_11', 'h_13', 'h_2', 'h_10', 'h_12', 'h_14'],
    'G3_h': ['h_4', 'h_12', 'h_14', 'h_23', 'h_13', 'h_15', 'h_20'],
    'TG_h': ['h_20', 'h_9'],
    'GG_m': ['m_7', 'm_8'],
    'G6_m': ['m_1', 'm_8', 'm_7', 'm_9'],
    'G3_m': ['m_9', 'm_10'],
    'TG_pl': ['j_0', 'h_9'],
    'Pyr_a': ['a_8', 'a_12', 'a_19', 'a_10', 'a_11'],
    'Ac_CoA_a': ['a_10', 'a_14', 'a_18', 'a_13', 'a_16'],
    'FA_CoA_a': ['a_2', 'a_13', 'a_7'],
    'Cit_a': ['a_16', 'a_14', 'a_15'],
    'OAA_a': ['a_11', 'a_14', 'a_15', 'a_17', 'a_9', 'a_12', 'a_16'],
    'NADPH_a': ['a_6', 'a_12', 'a_13'],
    'Pyr_h': ['h_5', 'h_15', 'h_24', 'h_29', 'h_16', 'h_25'],
    'Ac_CoA_h': ['h_16', 'h_18', 'h_26', 'h_27', 'h_17', 'h_19', 'h_6'],
    'FA_CoA_h': ['h_8', 'h_19', 'h_18', 'h_20'],
    'MVA_h': ['h_17', 'h_7'],
    'OAA_h': ['h_22', 'h_25', 'h_26', 'h_28', 'h_21', 'h_23', 'h_24'],
    'Cit_h': ['h_21', 'h_22', 'h_26'],
    'AA_h': ['h_1', 'h_27', 'h_28', 'h_29'],
    'NADPH_h': ['h_14', 'h_24', 'h_19'],
    'Pyr_m': ['m_10', 'm_17', 'm_11', 'm_2'],
    'Ac_CoA_m': ['m_3', 'm_11', 'm_12', 'm_18', 'm_13'],
    'FA_CoA_m': ['m_4', 'm_12'],
    'AA_m': ['m_5', 'm_21', 'm_6', 'm_17', 'm_18', 'm_19', 'm_20'],
    'Cit_m': ['m_13', 'm_14'],
    'OAA_m': ['m_14', 'm_19', 'm_13'],
    'H_cyt_m': ['m_10', 'm_15', 'm_2'],
    'H_mit_m': ['m_3', 'm_12', 'm_15', 'm_16'],
    'CO2_m': ['m_11', 'm_14'],
    'H2O_m': ['m_16'],
    'ATP_cyt_m': ['m_10'],
    'ATP_mit_m': ['m_16'],
    'Glu_ef': ['h_2', 'm_1', 'a_4', 'h_3', 'j_1'],
    'AA_ef': ['m_6', 'm_5', 'a_1', 'h_1', 'j_4'],
    'FFA_ef': ['j_0', 'a_3', 'm_4', 'a_2', 'h_8', 'j_3'],
    'KB_ef': ['m_3', 'j_2', 'h_6'],
    'Glycerol_ef': ['j_0', 'a_3', 'h_4'],
    'Lac_m': ['m_2', 'h_5'],
    'Urea_ef': ['j_4', 'a_17', 'a_18', 'a_19', 'm_17', 'm_18', 'm_19', 'h_27', 'h_28', 'h_29'],
    'Cholesterol_pl': ['h_7'],
    'INS': [],
    'GLN': [],
    'CAM': [],
    'Muscle_m': ['m_20', 'm_21']
}
