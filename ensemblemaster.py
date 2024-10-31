# This program run multiple WE simulation on multiple networks with different parameters

import os
import numpy as np
import const_xstar

if __name__ == '__main__':

    # Netwrok parameters

    # N = [300,400,500,600,700,800,900,1000,1100,1200,1300,1400]
    N = 5000
    prog = 'ig'
    lam = 1.3
    # lam = 1+np.logspace(-2,0,9)
    # lam = np.array([1.5,1.6,1.7,1.8])
    eps_din = 0.8
    eps_dout = 0.8
    #eps_din = [0.01,0.04,0.06,0.08,0.1,0.14,0.18,0.2,0.25,0.3,0.4,0.5,0.6]
    #eps_dout = [0.01,0.04,0.06,0.08,0.1,0.14,0.18,0.2,0.25,0.3,0.4,0.5,0.6]
    # correlation = [-0.01,-0.03,-0.05,-0.08,-0.1,-0.12,-0.15,-0.18,-0.2,-0.25,-0.3]
    correlation = 1.0
    number_of_networks = 20
    # k = [50]
    k= 50

    num_points_alpha_eps_xstar,fraction = 20,0.9
    alpha,epsilon = np.linspace(-correlation,correlation,num_points_alpha_eps_xstar),np.linspace(0.01,eps_din,num_points_alpha_eps_xstar)
    alpha_filtered, epsilon_filtered = const_xstar.find_constant_infected_fraction(lam, alpha, epsilon, fraction,k,prog,N)


    # We simulation parameters

    sims = 500
    tau = 0.5
    # tau = np.linspace(0.1,2.0,20)
    it = 70
    jump = 1
    new_trajectory_bin = 2
    error_graphs = False

    # Parameters that don't change

    relaxation_time = 20
    x = 0.2
    Alpha = 1.0
    run_mc_simulation = False
    short_path = False

    # Paths needed to run the program
    dir_path = os.path.dirname(os.path.realpath(__file__))
    slurm_path = dir_path +'/slurm.serjob python3'
    program_path = dir_path +'/runwesim.py'
    loop_over = correlation

    for i,j in zip(alpha_filtered,epsilon_filtered):
        error_graphs_flag = '--error_graphs' if error_graphs else ''
        run_mc_simulation_flag = '--run_mc_simulation' if run_mc_simulation else ''
        short_flag_flag = False
        command = (f'{slurm_path} {program_path} --N {N} --prog {prog} --lam {lam} --eps_din {j} '
                   f'--eps_dout {j} --correlation {i} --number_of_networks {number_of_networks} '
                   f'--k {k} {error_graphs_flag} --sims {sims} --tau {tau} --it {it} --jump {jump} '
                   f'--new_trajectory_bin {new_trajectory_bin} --relaxation_time {relaxation_time} --x {x} '
                   f'--Alpha {Alpha} {run_mc_simulation_flag}')
        os.system(command)

