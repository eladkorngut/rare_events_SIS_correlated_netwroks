import numpy as np
import os
import rand_networks
import csv
import pickle
import networkx as nx
from scipy.stats import skew
from scipy.sparse.linalg import eigsh
import netinithomo
import argparse


def export_parameters_to_csv(parameters,network_number):
    name_parameters = 'cparameters_{}.txt'.format(network_number)
    # N, sims, it, k, x, lam, jump, Alpha,Beta,number_of_networks, tau, mf_solution ,eps_din, eps_dout, new_trajcetory_bin, prog, Beta_avg,dir_path = parameters
    # cparameters=[N, sims, it, k, x, lam, jump, Alpha,Beta,number_of_networks, tau, mf_solution ,new_trajcetory_bin, prog, Beta_avg,dir_path]
    f =open(name_parameters,'+a')
    with f:
        writer = csv.writer(f)
        writer.writerow(parameters)
    f.close()

def export_network_to_csv(G,netname):
    # Open a CSV file for writing incoming neighbors

    # Check if the graph is directed
    is_directed = G.is_directed()

    with open('Adjin_{}.txt'.format(netname), 'w', newline='') as incoming_file:
        # Create a CSV writer
        incoming_writer = csv.writer(incoming_file)
        # Iterate over all nodes in the graph
        for node in np.sort(G):
            # Get the incoming neighbors of the current node
            if is_directed:
                incoming_neighbors = list(G.predecessors(node))
                # Get the degree of the current node
                degree = G.in_degree[node]
            else:
                incoming_neighbors = list(G.neighbors(node))  # All neighbors for undirected graph
                degree = G.degree[node]
            # Write a row to the CSV file for the current node
            joint = np.concatenate(([degree],incoming_neighbors),axis=0)
            incoming_writer.writerow(joint)
            # incoming_writer.writerow([degree])
            # for node in incoming_neighbors: incoming_writer.writerow([node])
    with open('Adjout_{}.txt'.format(netname), 'w', newline='') as outgoing_file:
        # Create a CSV writer
        outgoing_writer = csv.writer(outgoing_file)
        # Iterate over all nodes in the graph
        for node in np.sort(G):
            # Get the incoming neighbors of the current node
            if is_directed:
                outgoing_neighbors = list(G.successors(node))
                # Get the degree of the current node
                degree = G.out_degree[node]
            else:
                outgoing_neighbors = list(G.neighbors(node))  # All neighbors for undirected graph
                degree = G.degree[node]
            # Write a row to the CSV file for the current node
            joint = np.concatenate(([degree],outgoing_neighbors),axis=0)
            outgoing_writer.writerow(joint)


def act_as_main(foldername,parameters,Istar,prog):
    # This program will run the we_sis_network_extinction.py on the laptop\desktop
    # This function submit jobs to the cluster with the following program keys:
    # bd: creates a bimodal directed networks and find its mean time to extinction
    dir_path = os.path.dirname(os.path.realpath(__file__))
    slurm_path = dir_path +'/slurm.serjob'
    program_path = dir_path +'/cwesis.exe'
    os.mkdir(foldername)
    os.chdir(foldername)
    data_path = os.getcwd() +'/'
    if (prog=='pl'):
        N, sims, it, k, x, lam, jump, Num_inf, Alpha, number_of_networks, tau, a, new_trajcetory_bin,prog, Beta_avg,error_graphs = parameters
        N, sims, it, k, x, lam, jump, Num_inf, Alpha, number_of_networks, tau, a, new_trajcetory_bin, prog, Beta_avg,error_graphs=\
        int(N), int(sims), int(it), float(k), float(x), float(lam), int(jump), int(Num_inf), float(Alpha), int(number_of_networks), float(tau),\
        float(a), float(new_trajcetory_bin),prog, float(Beta_avg),bool(error_graphs)
        a_graph, b_graph = rand_networks.find_b_binary_search(float(k), int(N), float(a))
        if error_graphs==False:
            G = rand_networks.configuration_model_powerlaw(a_graph, b_graph, int(N))
            k_avg_graph = np.mean([G.degree(n) for n in G.nodes()])
            while (np.abs(k_avg_graph - float(k)) / float(k) > 0.05):
                if a < 5.0:
                    a_graph, b_graph = rand_networks.find_b_binary_search(float(k), int(N), float(a))
                else:
                    a_graph, b_graph = rand_networks.find_a_binary_search(float(k), int(N), float(a))
                G, a_graph, b_graph = rand_networks.configuration_model_powerlaw(a_graph, b_graph, int(N))
                k_avg_graph = np.mean([G.degree(n) for n in G.nodes()])
            Beta_graph = float(lam) / k_avg_graph
            eps_graph = np.std([G.degree(n) for n in G.nodes()]) / k_avg_graph
            Beta = Beta_graph / (1 + eps_graph ** 2)
    else:
        N,sims,it,k,x,lam,jump,Num_inf,Alpha,number_of_networks,tau,eps_din,eps_dout,new_trajcetory_bin,prog,Beta_avg,error_graphs = parameters
        N, sims, it, k, x, lam, jump, Num_inf, Alpha, number_of_networks, tau, eps_din, eps_dout, new_trajcetory_bin, prog, Beta_avg,error_graphs=\
        int(N),int(sims),int(it),float(k),float(x),float(lam),float(jump),int(Num_inf),float(Alpha),int(number_of_networks),float(tau),float(eps_din),float(eps_dout),\
        int(new_trajcetory_bin),prog,float(Beta_avg),bool(error_graphs)
        if error_graphs==True:
            G = rand_networks.configuration_model_undirected_graph_mulit_type(float(k),float(eps_din),int(N),prog)
            graph_degrees = np.array([G.degree(n) for n in G.nodes()])
            k_avg_graph,graph_std,graph_skewness = np.mean(graph_degrees),np.std(graph_degrees),skew(graph_degrees)
            second_moment,third_moment = np.mean((graph_degrees)**2),np.mean((graph_degrees)**3)
            eps_graph = graph_std / k_avg_graph
            # third_moment = graph_skewness * (graph_std ** 3)
            Beta_graph = float(lam)/k_avg_graph
            Beta = Beta_graph / (1 + eps_graph ** 2)
    if prog == 'bd':
        # G = nx.complete_graph(N)
        d1_in, d1_out, d2_in, d2_out = int(int(k) * (1 - float(eps_din))), int(int(k) * (1 - float(eps_dout))), int(int(k) * (1 + float(eps_din))), int(
            int(k) * (1 + float(eps_dout)))
        Beta = float(Beta_avg) / (1 + float(eps_din) * float(eps_dout))  # This is so networks with different std will have the reproduction number
        parameters = np.array([N,sims,it,k,x,lam,jump,Alpha,Beta,number_of_networks,tau,Istar,new_trajcetory_bin,dir_path,prog,dir_path,eps_din,eps_dout])
        np.save('parameters.npy',parameters)
    for i in range(int(number_of_networks)):
        if prog=='bd':
            G = rand_networks.random_bimodal_directed_graph(int(d1_in), int(d1_out), int(d2_in), int(d2_out), int(N))
            parameters = np.array([N,sims,it,k,x,lam,jump,Alpha,Beta,i,tau,Istar,new_trajcetory_bin,dir_path,prog,dir_path,eps_din,eps_dout])
        elif prog=='h':
            G = nx.random_regular_graph(int(k), int(N))
            parameters = np.array([N,sims,it,k,x,lam,jump,Alpha,Beta_avg,i,tau,Istar,new_trajcetory_bin,dir_path,prog,dir_path,eps_din,eps_dout])            # Creates a random graphs with k number of neighbors
        elif prog == 'pl':
            G = rand_networks.configuration_model_powerlaw(a_graph, b_graph, int(N))
            k_avg_graph = np.mean([G.degree(n) for n in G.nodes()])
            while (np.abs(k_avg_graph - float(k)) / float(k) > 0.05):
                if error_graphs==False:
                    if a < 5.0:
                        a_graph, b_graph = rand_networks.find_b_binary_search(float(k), int(N), float(a))
                    else:
                        a_graph, b_graph = rand_networks.find_a_binary_search(float(k), int(N), float(a))
                    G, a_graph, b_graph = rand_networks.configuration_model_powerlaw(a_graph, b_graph, int(N))
                    k_avg_graph = np.mean([G.degree(n) for n in G.nodes()])
                Beta_graph = float(lam) / k_avg_graph
                eps_graph = np.std([G.degree(n) for n in G.nodes()]) / k_avg_graph
                Beta = Beta_graph / (1 + eps_graph ** 2)
            parameters = np.array(
                [N, sims, it, k_avg_graph, x, lam, jump, Alpha, Beta, i, tau, Istar, new_trajcetory_bin, prog, data_path,
                 eps_graph, eps_graph, a_graph, b_graph])
            np.save('parameters_{}.npy'.format(i), parameters)
        elif prog=='exp':
            G = rand_networks.configuration_model_undirected_graph_exp(float(k), int(N))
            graph_degrees = np.array([G.degree(n) for n in G.nodes()])
            k_avg_graph,graph_std,graph_skewness = np.mean(graph_degrees),np.std(graph_degrees),skew(graph_degrees)
            second_moment,third_moment = np.mean((graph_degrees)**2),np.mean((graph_degrees)**3)
            eps_graph = graph_std / k_avg_graph
            # third_moment = graph_skewness * (graph_std ** 3)
            Beta_graph = float(lam)/k_avg_graph
            Beta = Beta_graph / (1 + eps_graph ** 2)
            parameters = np.array([N,sims,it,k_avg_graph,x,lam,jump,Alpha,Beta,i,tau,Istar,new_trajcetory_bin,
                                   dir_path,prog,dir_path,eps_graph,eps_graph,graph_std,graph_skewness,third_moment,second_moment])
            np.save('parameters_{}.npy'.format(i), parameters)
        else:
            if error_graphs==False:
                G = rand_networks.configuration_model_undirected_graph_mulit_type(float(k),float(eps_din),int(N),prog)
                graph_degrees = np.array([G.degree(n) for n in G.nodes()])
                k_avg_graph, graph_std, graph_skewness = np.mean(graph_degrees), np.std(graph_degrees), skew(
                    graph_degrees)
                second_moment,third_moment = np.mean((graph_degrees)**2),np.mean((graph_degrees)**3)
                eps_graph = graph_std / k_avg_graph
                # third_moment = graph_skewness * (graph_std ** 3)
                Beta_graph = float(lam)/k_avg_graph
                Beta = Beta_graph / (1 + eps_graph ** 2)
            parameters = np.array([N,sims,it,k_avg_graph,x,lam,jump,Alpha,Beta,i,tau,Istar,new_trajcetory_bin,dir_path,
                                   prog,dir_path,eps_graph,eps_graph,graph_std,graph_skewness,third_moment,second_moment])
            np.save('parameters_{}.npy'.format(i), parameters)
        infile = 'GNull_{}.pickle'.format(i)
        with open(infile,'wb') as f:
            pickle.dump(G,f,pickle.HIGHEST_PROTOCOL)
        # nx.write_gpickle(G, infile)
        export_network_to_csv(G, i)
        export_parameters_to_csv(parameters,i)
        path_adj_in = data_path + 'Adjin_{}.txt'.format(i)
        path_adj_out = data_path + 'Adjout_{}.txt'.format(i)
        path_parameters = data_path + 'cparameters_{}.txt'.format(i)
        parameters_path ='{} {} {}'.format(path_adj_in,path_adj_out,path_parameters)


def job_to_cluster(foldername,parameters,Istar,error_graphs,run_mc_simulation,short_path):
    # This function submit jobs to the cluster with the following program keys:
    # bd: bimodal network, h:homogenous, exp:exponential, gam:gamma, bet:beta, ln:log-normal, ig:Wald

    dir_path = os.path.dirname(os.path.realpath(__file__))
    slurm_path = dir_path +'/slurm.serjob'
    program_path = dir_path +'/cwesis.exe'
    os.mkdir(foldername)
    os.chdir(foldername)
    data_path = os.getcwd() +'/'
    N, sims, it, k, x, lam, jump, Num_inf, Alpha, number_of_networks, tau, eps_din, eps_dout, new_trajcetory_bin, prog, Beta_avg, error_graphs_parmeters,correlation = parameters
    N, sims, it, k, x, lam, jump, Num_inf, Alpha, number_of_networks, tau, eps_din, eps_dout, new_trajcetory_bin, prog, Beta_avg,correlation = \
        int(N), int(sims), int(it), float(k), float(x), float(lam), float(jump), int(Num_inf), float(Alpha), int(number_of_networks),\
        float(tau), float(eps_din), float(eps_dout),int(new_trajcetory_bin), prog, float(Beta_avg), float(correlation)
    for i in range(int(number_of_networks)):


        if error_graphs==False:
            G, graph_degrees = rand_networks.configuration_model_undirected_graph_mulit_type(float(k), float(eps_din),int(N), prog,correlation)
            k_avg_graph, graph_std, graph_skewness = np.mean(graph_degrees), np.std(graph_degrees), skew(graph_degrees)
            second_moment, third_moment = np.mean((graph_degrees) ** 2), np.mean((graph_degrees) ** 3)
            eps_graph = graph_std / k_avg_graph
            largest_eigenvalue,largest_eigen_vector = eigsh(nx.adjacency_matrix(G).astype(float), k=1, which='LA', return_eigenvectors=True)
            mean_shortest_path_length = nx.average_shortest_path_length(G) if short_path==True else 0
            Beta = float(lam) / largest_eigenvalue[0]
            graph_correlation = nx.degree_assortativity_coefficient(G)
            rho = (np.sum(largest_eigen_vector) / (N * np.sum(largest_eigen_vector ** 3))) * (Beta * largest_eigenvalue[0] - 1)
            parameters = np.array(
                [N, sims, it, k_avg_graph, x, lam, jump, Alpha, Beta, i, tau, Istar, new_trajcetory_bin, dir_path,
                 prog, eps_graph, eps_graph, graph_std, graph_skewness, third_moment, second_moment,graph_correlation,rho])
        np.save('parameters_{}.npy'.format(i), parameters)
        np.save('largest_eigen_vector_{}.npy'.format(i), largest_eigen_vector)
        np.save('largest_eigenvalue_{}.npy'.format(i), largest_eigenvalue[0])
        np.save(f'mean_shortest_path_length_{i}.npy',mean_shortest_path_length)
        infile = 'GNull_{}.pickle'.format(i)
        with open(infile,'wb') as f:
            pickle.dump(G,f,pickle.HIGHEST_PROTOCOL)
        export_network_to_csv(G, i)
        export_parameters_to_csv(parameters,i)
        path_adj_in = data_path + 'Adjin_{}.txt'.format(i)
        path_adj_out = data_path + 'Adjout_{}.txt'.format(i)
        path_parameters = data_path + 'cparameters_{}.txt'.format(i)
        parameters_path ='{} {} {}'.format(path_adj_in,path_adj_out,path_parameters)
        os.system('{} {} {}'.format(slurm_path,program_path,parameters_path))
        if run_mc_simulation==True:
            # Convert the undirected graph to a directed graph with bidirectional edges
            G = G.to_directed()
            G = netinithomo.set_graph_attriubute_DiGraph(G)
            with open(infile, 'wb') as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
            prog_mc = 'gam'
            bank,Num_inital_conditions = 1000000,100
            outfile ='mc_N_{}_eps_{}_R_{}'.format(N,eps_din,lam)
            os.system(dir_path + '/slurm.serjob python3 ' + dir_path + '/gillespierunhomo.py ' + str(prog_mc) + ' ' +
                      str(Alpha) + ' ' + str(bank) + ' ' + str(outfile) + ' ' + str(infile) + ' ' + str(
                Num_inital_conditions) + ' ' + str(Num_inf) + ' ' + str(i) + ' ' + str(Beta))

        # os.system('{} {} {} {}'.format(program_path,path_adj_in,path_adj_out,path_parameters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process network and WE method parameters.")

    # Parameters for the network
    parser.add_argument('--N', type=int, help='Number of nodes')
    parser.add_argument('--prog', type=str, help='Program')
    parser.add_argument('--lam', type=float, help='The reproduction number')
    parser.add_argument('--eps_din', type=float, help='The normalized std (second moment divided by the first) of the in-degree distribution')
    parser.add_argument('--eps_dout', type=float, help='The normalized std (second moment divided by the first) of the out-degree distribution')
    parser.add_argument('--correlation', type=float, help='Correlation parameter')
    parser.add_argument('--number_of_networks', type=int, help='Number of networks')
    parser.add_argument('--k', type=int, help='Average number of neighbors for each node')
    parser.add_argument('--error_graphs', action='store_true', help='Flag for error graphs')

    # Parameters for the WE method
    parser.add_argument('--sims', type=int, help='Number of simulations at each bin')
    parser.add_argument('--tau', type=float, help='Tau parameter')
    parser.add_argument('--it', type=int, help='Number of iterations')
    parser.add_argument('--jump', type=int, help='Jump parameter')
    parser.add_argument('--new_trajectory_bin', type=int, help='New trajectory bin')

    # Parameters that don't get changed
    parser.add_argument('--relaxation_time', type=int, help='Relaxation time')
    parser.add_argument('--x', type=float, help='Initial infection percentage')
    parser.add_argument('--Alpha', type=float, help='Recovery rate')
    parser.add_argument('--run_mc_simulation', action='store_true', help='Flag to run MC simulation')
    parser.add_argument('--short_path', action='store_true', help='Flag to measure mean shortest path')

    args = parser.parse_args()

    # Default parameters
    N = 5000 if args.N is None else args.N
    prog = 'gam' if args.prog is None else args.prog
    lam = 1.3 if args.lam is None else args.lam
    eps_din = 0.3 if args.eps_din is None else args.eps_din
    eps_dout = 0.3 if args.eps_dout is None else args.eps_dout
    correlation = 0.7 if args.correlation is None else args.correlation
    number_of_networks = 5 if args.number_of_networks is None else args.number_of_networks
    k = 50 if args.k is None else args.k
    error_graphs = args.error_graphs

    sims = 500 if args.sims is None else args.sims
    tau = 1.0 if args.tau is None else args.tau
    it = 70 if args.it is None else args.it
    jump = 1 if args.jump is None else args.jump
    new_trajectory_bin = 2 if args.new_trajectory_bin is None else args.new_trajectory_bin

    relaxation_time = 20 if args.relaxation_time is None else args.relaxation_time
    x = 0.2 if args.x is None else args.x
    Num_inf = int(x * N)
    Alpha = 1.0 if args.Alpha is None else args.Alpha
    Beta_avg = Alpha * lam / k
    run_mc_simulation = args.run_mc_simulation
    # run_mc_simulationtion = True
    short_path = False


    parameters = np.array([N, sims, it, k, x, lam, jump, Num_inf, Alpha, number_of_networks, tau, eps_din, eps_dout, new_trajectory_bin, prog, Beta_avg, error_graphs, correlation])
    graphname = 'GNull'
    foldername = 'prog_{}_N{}_k_{}_R_{}_tau_{}_it_{}_jump_{}_new_trajectory_bin_{}_sims_{}_net_{}_epsin_{}_epsout_{}_correlation_{}_err_{}'.format(
        prog, N, k, lam, tau, it, jump, new_trajectory_bin, sims, number_of_networks, eps_din, eps_dout, correlation, error_graphs)
    Istar = (1 - 1/lam) * N

    job_to_cluster(foldername, parameters, Istar, error_graphs, run_mc_simulation,short_path)
    # act_as_main(foldername, parameters, Istar, prog)

