from numba import njit

import default as d


@njit
def calculate_bmr(
    INS,
    J_prot_flow, J_carb_flow,
):
    need_to_fill_bmr = (1 - d.BASE_AA_BMR - d.BASE_GLU_BMR) * d.BMR_ON_GRID

    bmr_AA_ef = d.BASE_AA_BMR_VALUE
    bmr_Glu_ef = d.BASE_GLU_BMR_VALUE
    bmr_KB_ef = d.BASE_KB_BMR_VALUE
    bmr_FFA_ef = 0.0
    
    if J_prot_flow > 0:
        if J_prot_flow / (need_to_fill_bmr * 1/d.beta_AA_ef) > 1.0:
            bmr_AA_ef += need_to_fill_bmr * 1/d.beta_AA_ef
            return bmr_AA_ef, bmr_Glu_ef, bmr_FFA_ef, bmr_KB_ef
        else:
            bmr_AA_ef += J_prot_flow
            need_to_fill_bmr -= J_prot_flow

    if J_carb_flow > 0 and INS > 0:
        if J_carb_flow / (need_to_fill_bmr * 1/d.beta_Glu_ef) > 1.0:
            bmr_Glu_ef += need_to_fill_bmr * 1/d.beta_Glu_ef
            return bmr_AA_ef, bmr_Glu_ef, bmr_FFA_ef, bmr_KB_ef
        else:
            bmr_Glu_ef += J_carb_flow
            need_to_fill_bmr -= J_carb_flow

    bmr_FFA_ef += need_to_fill_bmr * 1/d.beta_FFA_ef
    return bmr_AA_ef, bmr_Glu_ef, bmr_FFA_ef, bmr_KB_ef

    


# @njit
# def calculate_bmr(AA_ef, Glu_ef, TG_pl):
#     need_to_fill_bmr = d.BMR_ON_GRID

#     bmr_AA_ef = 0.0
#     bmr_Glu_ef = 0.0
#     bmr_TG_pl = 0.0

#     if AA_ef  - AA_ef* need_to_fill_bmr * 1/d.beta_AA_ef > 10.0:
#         bmr_AA_ef = need_to_fill_bmr * 1/d.beta_AA_ef
#         return bmr_AA_ef, bmr_Glu_ef, bmr_TG_pl
#     elif AA_ef < 10.0:
#         bmr_AA_ef = 0.0
#     else:
#         coeff = (AA_ef - 10.0) / (AA_ef * 1/d.beta_AA_ef)
#         need_to_fill_bmr -= coeff
#         bmr_AA_ef = coeff

#     if Glu_ef  - Glu_ef* need_to_fill_bmr * 1/d.beta_Glu_ef  > 5:
#         bmr_Glu_ef = need_to_fill_bmr * 1/d.beta_Glu_ef
#         return bmr_AA_ef, bmr_Glu_ef, bmr_TG_pl
#     if Glu_ef < 5:
#         bmr_Glu_ef = 0.0
#     else:
#         coeff = (Glu_ef - 5) / (Glu_ef * 1/d.beta_Glu_ef)
#         need_to_fill_bmr -= coeff
#         bmr_Glu_ef  = coeff

#     bmr_TG_pl = need_to_fill_bmr * 1/d.beta_TG_pl
#     return bmr_AA_ef, bmr_Glu_ef, bmr_TG_pl

