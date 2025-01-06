# This program run multiple WE simulation on multiple networks with different parameters

import os
import numpy as np

if __name__ == '__main__':

    # Netwrok parameters

    # N = [300,400,500,600,700,800,900,1000,1100,1200,1300,1400]
    N = 5000
    prog = 'pgp'
    # lam = 1.3
    # lam = 1+np.logspace(-2,0,9)
    lam = [1.1,1.2,1.3,1.4,1.5]
    measurements = 50
    eps_din = 0.1
    # eps_din = np.random.uniform(0.0, 3.0,measurements)
    eps_dout = eps_din
    # eps_din = [0.01,0.04,0.06,0.08,0.1,0.14,0.18,0.2,0.25,0.3,0.4,0.5,0.6]
    # eps_dout = [0.01,0.04,0.06,0.08,0.1,0.14,0.18,0.2,0.25,0.3,0.4,0.5,0.6]
    # correlation = [-0.01,-0.03,-0.05,-0.08,-0.1,-0.12,-0.15,-0.18,-0.2,-0.25,-0.3]
    # correlation = np.random.uniform(-0.6, 0.6,measurements)
    correlation = 0.0
    number_of_networks = 1
    # k = [50]
    k= 50

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
    slurm_path = dir_path + '/slurm.serjob python3'
    program_path = dir_path + '/runwesim.py'
    loop_over = lam

    for i in loop_over:
        error_graphs_flag = '--error_graphs' if error_graphs else ''
        run_mc_simulation_flag = '--run_mc_simulation' if run_mc_simulation else ''
        short_flag_flag = False
        command = (f'{slurm_path} {program_path} --N {N} --prog {prog} --lam {i} --eps_din {eps_din} '
                   f'--eps_dout {eps_din} --correlation {correlation} --number_of_networks {number_of_networks} '
                   f'--k {k} {error_graphs_flag} --sims {sims} --tau {tau} --it {it} --jump {jump} '
                   f'--new_trajectory_bin {new_trajectory_bin} --relaxation_time {relaxation_time} --x {x} '
                   f'--Alpha {Alpha} {run_mc_simulation_flag}')
        os.system(command)