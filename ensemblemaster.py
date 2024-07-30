# This program run multiple WE simulation on multiple networks with different parameters

import os

if __name__ == '__main__':

    # Netwrok parameters
    N = 7500
    prog = 'bd'
    lam = 1.3
    eps_din = 0.02
    eps_dout = 0.02
    correlation = [0.3,0.4]
    number_of_networks = 5
    k = 50

    # We simulation parameters
    sims = 1000
    tau = 0.4
    it = 70
    jump = 1
    new_trajectory_bin = 2
    error_graphs = False

    # Parameters that don't change
    relaxation_time = 20
    x = 0.2
    Num_inf = int(x * N)
    Alpha = 1.0
    Beta_avg = Alpha * lam / k
    run_mc_simulation = False

    # Paths needed to run the program
    dir_path = os.path.dirname(os.path.realpath(__file__))
    slurm_path = dir_path +'/slurm.serjob python3'
    program_path = dir_path +'/runwesim.py'
    loop_over = correlation

    for i in correlation:
        error_graphs_flag = '--error_graphs' if error_graphs else ''
        run_mc_simulation_flag = '--run_mc_simulation' if run_mc_simulation else ''

        command = (f'{slurm_path} {program_path} --N {N} --prog {prog} --lam {lam} --eps_din {eps_din} '
                   f'--eps_dout {eps_dout} --correlation {i} --number_of_networks {number_of_networks} '
                   f'--k {k} {error_graphs_flag} --sims {sims} --tau {tau} --it {it} --jump {jump} '
                   f'--new_trajectory_bin {new_trajectory_bin} --relaxation_time {relaxation_time} --x {x} '
                   f'--Alpha {Alpha} {run_mc_simulation_flag}')



