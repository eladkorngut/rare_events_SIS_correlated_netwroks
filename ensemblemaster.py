import os



if __name__ == '__main__':
    N = 7500
    prog = 'bd'
    lam = 1.3
    eps_din = 0.02
    eps_dout = 0.02
    correlation = [0.3,0.4]
    number_of_networks = 5
    k = 50
    error_graphs = False

    sims = 1000
    tau = 0.4
    it = 70
    jump = 1
    new_trajectory_bin = 2

    relaxation_time = 20
    x = 0.2
    Num_inf = int(x * N)
    Alpha = 1.0
    Beta_avg = Alpha * lam / k
    run_mc_simulation = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    slurm_path = dir_path +'/slurm.serjob python3'
    program_path = dir_path +'/runwesim.py'
    loop_over = correlation

    for i in correlation:
        os.system(f'{slurm_path} {program_path} {N} {prog} {lam} {eps_din} {eps_dout} {i} '
                  f'{number_of_networks} {k} {error_graphs} {sims} {tau} {it} {jump} {new_trajectory_bin} '
                  f'{relaxation_time} {x} {Alpha} {run_mc_simulation}')


