import numpy as np
from scipy.integrate import ode

from input import *
from Plotting.myplt import *
from myo_supportfs import *
from coefficient_controller import CoefficientsController
from equations import EquationsController
from default import *


def make_equations_and_coefficient_controllers():
    coefficients = make_default_coefficients()
    coefficient_controller = CoefficientsController(coefficients)
    equations_controller = EquationsController(coefficient_controller)
    return coefficient_controller, equations_controller


class CalculationModel:

    def __init__(self) -> None:
        coefficient_controller, equations_controller = make_equations_and_coefficient_controllers()
        self.coefficient_controller = coefficient_controller
        self.equations_controller = equations_controller
        self.start_point_dict = start_point_dict

    def change_start_point(self, points: dict[str, float]):
        self.start_point_dict.update(points)

    def update_base_coefficient_value(self, coefficients: dict[str, float]):
        """
        метод для изменения значения коэффициента
        """
        self.coefficient_controller.update_base_coefficient_value(coefficients=coefficients)

    def calculate(self):
        """
        основная функция, которая запускает всве процессы по расчету уравнений, коэффициентов и построению графиков
        """

        index_by_name, name_by_index, start_point = get_start_point_names_mapping(self.start_point_dict)

        J_flow_carb_vs = J_flow_carb_func.values
        J_flow_prot_vs = J_flow_prot_func.values
        J_flow_fat_vs  = J_flow_fat_func.values 

        # AUC auxiliary arrays
        INS_on_grid = np.zeros(shape=(len(time_grid), ),dtype=np.float32)
        INS_AUC_w_on_grid = np.zeros(shape=(len(time_grid), ),dtype=np.float32)
        INS_on_grid[0] = start_point[index_by_name['INS']]
        INS_AUC_w_on_grid[0] = 0.0  
        T_a_on_grid = np.zeros(shape=(len(time_grid), ),dtype=np.float32)
        T_a_on_grid[0]= 0.0
        last_seen_time = np.zeros(shape=(1,),dtype=np.float32)
        last_seen_time[0] = t_0
        last_time_pos = np.zeros(shape=(1,),dtype=np.intc)
        last_time_pos[0] = 0


        def F_wrapped(t, y):
            return self.equations_controller.calculate_equations(
                t,y,INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid,last_seen_time,last_time_pos,
                J_flow_carb_vs,
                J_flow_prot_vs,
                J_flow_fat_vs,

                myocyte_coefficients_base=self.coefficient_controller.myocyte_coefficients_base,
                adipocyte_coefficients_base=self.coefficient_controller.adipocyte_coefficients_base,
                hepatocyte_coefficients_base=self.coefficient_controller.hepatocyte_coefficients_base,
                fluid_coefficients_base=self.coefficient_controller.fluid_coefficients_base,
            )

        solver = ode(f=F_wrapped)
        solver.set_initial_value(y=start_point,t=t_0)
        solver_type = 'vode'
        solver.set_integrator(solver_type, atol=1e-6, rtol=1e-4) 
        solutions = np.zeros(shape=(len(time_grid),len(start_point)),dtype=np.float32)
        solutions[0,:] = solver.y
        i_=  1
        print("STARTED")
        from time import time
        t1 = time()
        while solver.successful() and solver.t < t_end:
            solutions[i_,:] = solver.integrate(solver.t+tau_grid)
            i_ += 1
            if i_ % 20000 == 0:
                print(i_/432001)
        
        print("\n\n\n", "Время выполнения с j, функциями:", time() - t1,"\n\n\n")
        print('last solver time step {} target last step {}'.format(i_, len(time_grid)-1))
        time_sol = time_grid

        print(solutions.shape)
        print(time_sol.shape)    

        intervals = {}

        h_max = np.max(solutions)
        h_min = np.min(solutions)
        print(h_min,h_max)

        step_ = (h_max-h_min)/10

        fig = init_figure(x_label=r'$t,min$',y_label=r'$\frac{mmol}{L}$')
        fig = plot_solutions(fig, solutions, time_sol, name_by_index)

        add_line_to_fig(fig, time_grid, np.array([J_fat_func(t) for t in time_grid]), r'Fat')
        add_line_to_fig(fig, time_grid, np.array([J_prot_func(t) for t in time_grid]), r'Prot')
        add_line_to_fig(fig, time_grid, np.array([J_carb_func(t) for t in time_grid]), r'Carb')

        add_line_to_fig(fig, time_grid, np.array([J_flow_fat_func(t) for t in time_grid]), r'J_{TG}^{+}')
        add_line_to_fig(fig, time_grid, np.array([J_flow_prot_func(t) for t in time_grid]), r'J_{AA}^{+}')
        add_line_to_fig(fig, time_grid, np.array([J_flow_carb_func(t) for t in time_grid]), r'J_{Glu}^{+}')

        add_line_to_fig(fig, time_grid, T_a_on_grid, r'T_{a}')
        add_line_to_fig(fig, time_grid, INS_AUC_w_on_grid, r'AUC_{w}(INS)')



        add_line_to_fig(fig,time_sol, EnergyOnGrid(AA=solutions[:,index_by_name['AA_ef']],
                                                FFA=solutions[:,index_by_name['FFA_ef']],
                                                KB=solutions[:,index_by_name['KB_ef']],
                                                Glu=solutions[:,index_by_name['Glu_ef']],
                                                beta_AA=beta_AA_ef,beta_FFA=beta_FFA_ef,beta_KB=beta_KB_ef,beta_Glu=beta_Glu_ef),
                        r'E_{system}[kkal]')    


        fig = plot_intervals_to_plotly_fig(fig, intervals, 
                                        {    'INS': h_max-step_,
                                                'GLN_CAM': h_max-step_*2,
                                                'GLN_INS_CAM': h_max-step_*3,
                                                'fasting':h_max-step_*4},
                                        {    'INS': "#FF0000",
                                                'GLN_CAM': "#7FFF00",
                                                'GLN_INS_CAM': "#87CEEB",
                                                'fasting':"#04e022"})
        
        print('show fig1')


        fig2 = init_figure(r'$t,min$',y_label=r'$$')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['Glu_ef']], r'Glu_ef')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['AA_ef']], r'AA_ef')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['FFA_ef']], r'FFA_ef')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['KB_ef']], r'KB_ef')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['TG_a']], r'TG_a')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['AA_a']], r'AA_a')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['G6_a']], r'G6_a')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['G3_a']], r'G3_a')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['GG_h']], r'GG_h')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['G6_h']], r'G6_h')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['G3_h']], r'G3_h')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['TG_h']], r'TG_h')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['GG_m']], r'GG_m')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['G3_m']], r'G3_m')
        add_line_to_fig(fig2, time_sol, solutions[:,index_by_name['TG_pl']], r'TG_pl')
        print('show fig2')


        fig_a = init_figure(r'$t,min$',y_label=r'$$')
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["TG_a"]], r"TG_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["AA_a"]], r"AA_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["G6_a"]], r"G6_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["G3_a"]], r"G3_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["Pyr_a"]], r"Pyr_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["Ac_CoA_a"]], r"Ac_CoA_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["FA_CoA_a"]], r"FA_CoA_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["Cit_a"]], r"Cit_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["OAA_a"]], r"OAA_a")
        add_line_to_fig(fig_a, time_sol, solutions[:,index_by_name["NADPH_a"]], r"NADPH_a")
        print('show fig_a')

        fig_h = init_figure(r'$t,min$',y_label=r'$$')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['GG_h']], r'GG_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['G6_h']], r'G6_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['G3_h']], r'G3_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['TG_h']], r'TG_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['Pyr_h']], r'Pyr_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['MVA_h']], r'MVA_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['OAA_h']], r'OAA_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['Cit_h']], r'Cit_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['AA_h']], r'AA_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['NADPH_h']], r'NADPH_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['Ac_CoA_h']], r'Ac_CoA_h')
        add_line_to_fig(fig_h, time_sol, solutions[:,index_by_name['FA_CoA_h']], r'FA_CoA_h')
        print('show fig_h')

        fig_m = init_figure(r'$t,min$',y_label=r'$$')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['Muscle_m']], r'Muscle_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['AA_m']], r'AA_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['GG_m']], r'GG_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['G6_m']], r'G6_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['G3_m']], r'G3_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['Pyr_m']], r'Pyr_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['Cit_m']], r'Cit_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['OAA_m']], r'OAA_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['CO2_m']], r'CO2_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['H2O_m']], r'H2O_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['H_cyt_m']], r'H_cyt_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['H_mit_m']], r'H_mit_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['Ac_CoA_m']], r'Ac_CoA_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['FA_CoA_m']], r'FA_CoA_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['ATP_cyt_m']], r'ATP_cyt_m')
        add_line_to_fig(fig_m, time_sol, solutions[:,index_by_name['ATP_mit_m']], r'ATP_mit_m')
        print('show fig_m')


        fig_ef = init_figure(r'$t,min$',y_label=r'$$')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['Urea_ef']], r'Urea_ef')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['Glu_ef']], r'Glu_ef')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['AA_ef']], r'AA_ef')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['FFA_ef']], r'FFA_ef')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['KB_ef']], r'KB_ef')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['Glycerol_ef']], r'Glycerol_ef')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['Lac_m']], r'Lac_m')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['TG_pl']], r'TG_pl')
        add_line_to_fig(fig_ef, time_sol, solutions[:,index_by_name['Cholesterol_pl']], r'Cholesterol_pl')
        print('show fig_ef')

        fig.show()
        fig2.show()
        fig_a.show()
        fig_h.show()
        fig_m.show()
        fig_ef.show()
        return fig, fig2
    
CM = CalculationModel()
CM.calculate()