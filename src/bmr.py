import numpy as np
from numba import jit, njit

import default_my as d


@njit
def calculate_bmr(AA_ef, Glu_ef, TG_pl):
    need_to_fill_bmr = 7 * 10**(-2)

    bmr_AA_ef = 0.0
    bmr_Glu_ef = 0.0
    bmr_TG_pl = 0.0

    if AA_ef  - AA_ef* need_to_fill_bmr * 1/d.beta_AA_ef  > 10.0:
        bmr_AA_ef = need_to_fill_bmr * 1/d.beta_AA_ef
        return bmr_AA_ef, bmr_Glu_ef, bmr_TG_pl
    else:
        coeff = (AA_ef - 10.0) / (AA_ef * 1/d.beta_AA_ef)
        need_to_fill_bmr -= coeff
        bmr_AA_ef = coeff

    if Glu_ef  - Glu_ef* need_to_fill_bmr * 1/d.beta_Glu_ef  > 10.0:
        bmr_Glu_ef = need_to_fill_bmr * 1/d.beta_Glu_ef
        return bmr_AA_ef, bmr_Glu_ef, bmr_TG_pl
    else:
        coeff = (Glu_ef - 10.0) / (Glu_ef * 1/d.beta_Glu_ef)
        need_to_fill_bmr -= coeff
        bmr_Glu_ef  = coeff

    bmr_TG_pl = need_to_fill_bmr * 1/d.beta_TG_pl
    return bmr_AA_ef, bmr_Glu_ef, bmr_TG_pl

