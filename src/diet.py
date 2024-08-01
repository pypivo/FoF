import numpy as np
from input import *
import torch
from local_contributor_config import problem_folder


tau_grid = 0.1 # [min]
t_0 = 0.0 # [min]
t_end = 6000.0 # [min]


# make input data

diet_table_path = '/Users/hicebook/fof/fof2/FoF/input_data/diet_Mikhail.xlsx'

# считать начальлные данные по диете
diet_data = read_diet(diet_table_path)

# рассчет БЖУ
F_carb_chunks,_ = make_Fcarb(diet_data)
F_prot_chunks,_ = make_Fprot(diet_data)
F_fat_chunks,_  = make_Ffat(diet_data)

J_s_prot = J_sum(V_total=float(90/60),tau=tau_grid) # [ммоль/мин]
for i in range(len(F_prot_chunks)):
    ch_ = F_prot_chunks[i]
    J_s_prot.add_J_ch(ch_.t1,ch_.t2,delta_t = tau_grid,tau=30.0,T=480.0,rho=ch_.rho,alpha=10.0,volume=1.0)

J_s_fat = J_sum_with_infinit_v() # [ммоль/мин]
for i in range(len(F_prot_chunks)):
    ch_ = F_fat_chunks[i]
    J_s_fat.add_J_ch(ch_.t1,ch_.t2,delta_t = tau_grid,tau=30.0,T=300.0,rho=ch_.rho,alpha=1.239,volume=1.0)

J_s_carb = J_sum_with_infinit_v() # [ммоль/мин]
for i in range(len(F_prot_chunks)):
    ch_ = F_carb_chunks[i]
    J_s_carb.add_J_ch(ch_.t1,ch_.t2,delta_t = tau_grid,tau=30.0,T=120.0,rho=ch_.rho,alpha=5.55,volume=1.0)

N = int((t_end-t_0)/tau_grid)+1
time_grid = np.linspace(start=t_0, stop=t_end, num=N)
J_total_AA_ef = np.zeros(shape=(len(time_grid),),dtype=np.float32)
J_total_Glu_ef = np.zeros(shape=(len(time_grid),),dtype=np.float32)
J_total_TG_pl = np.zeros(shape=(len(time_grid),),dtype=np.float32)
ddt_AA_ef = np.zeros(shape=(len(time_grid),),dtype=np.float32)
ddt_Glu_ef = np.zeros(shape=(len(time_grid),),dtype=np.float32)
ddt_TG_pl = np.zeros(shape=(len(time_grid),),dtype=np.float32)
v_vec = np.zeros(shape=(len(time_grid),),dtype=np.float32)
n_vec = np.zeros(shape=(len(time_grid),),dtype=np.float32)
# fat_ch0 = np.zeros(shape=(len(time_grid),),dtype=np.float32)
for i in tqdm(range(len(time_grid))):
    # update values
    J_s_prot.step(time_grid[i])
    J_s_fat.step(time_grid[i])
    J_s_carb.step(time_grid[i])
    # get derivations
    J_total_AA_ef[i] = J_s_prot.get_J(time_grid[i])
    J_total_Glu_ef[i] = J_s_carb.get_J(time_grid[i])
    J_total_TG_pl[i] = J_s_fat.get_J(time_grid[i])
    ddt_AA_ef[i] = - J_s_prot.get_dJdt(time_grid[i])
    ddt_Glu_ef[i] = - J_s_carb.get_dJdt(time_grid[i])
    ddt_TG_pl[i] = - J_s_fat.get_dJdt(time_grid[i])
    # fat_ch0[i] = J_s_fat.J_arr[0].get_J()



fig,ax = plt.subplots(nrows=3,ncols=1)
fig.set_size_inches(10,9)
ax[0].plot(time_grid,ddt_AA_ef)
ax[1].plot(time_grid,ddt_Glu_ef)
ax[2].plot(time_grid,ddt_TG_pl)
fig2,ax2 = plt.subplots(nrows=3,ncols=1)
fig2.set_size_inches(10,9)
ax2[0].plot(time_grid,J_total_AA_ef)
ax2[0].set_title(r'$AA_{in}$')
ax2[1].plot(time_grid,J_total_Glu_ef)
ax2[1].set_title(r'$Glu_{in}$')
ax2[2].plot(time_grid,J_total_TG_pl)
ax2[2].set_title(r'$TG_{in}$')

torch.save(
    func_on_linear_grid(tau_grid,t_0,t_end, ddt_AA_ef),
    os.path.join(problem_folder, 'ddt_AA_ef'))
torch.save(
    func_on_linear_grid(tau_grid,t_0,t_end, ddt_Glu_ef), 
    os.path.join(problem_folder, 'ddt_Glu_ef'))
torch.save(
    func_on_linear_grid(tau_grid,t_0,t_end, ddt_TG_pl),
    os.path.join(problem_folder, 'ddt_TG_pl'))
torch.save(
    func_on_linear_grid(tau_grid,t_0,t_end, J_total_AA_ef), 
    os.path.join(problem_folder, 'J_prot'))
torch.save(
    func_on_linear_grid(tau_grid,t_0,t_end, J_total_Glu_ef), 
    os.path.join(problem_folder, 'J_carb'))
torch.save(
    func_on_linear_grid(tau_grid,t_0,t_end, J_total_TG_pl), 
    os.path.join(problem_folder, 'J_fat'))
