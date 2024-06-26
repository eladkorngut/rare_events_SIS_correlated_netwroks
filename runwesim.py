import numpy as np
import os
import rand_networks
import csv
import pickle
import networkx as nx
from scipy.stats import skew
from scipy.sparse.linalg import eigsh
import netinithomo


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


def job_to_cluster(foldername,parameters,Istar,error_graphs,run_mc_simulation):
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

            # Convert the undirected graph to a directed graph with bidirectional edges
            G = G.to_directed()
            G = netinithomo.set_graph_attriubute_DiGraph(G)

            k_avg_graph, graph_std, graph_skewness = np.mean(graph_degrees), np.std(graph_degrees), skew(graph_degrees)
            second_moment, third_moment = np.mean((graph_degrees) ** 2), np.mean((graph_degrees) ** 3)
            eps_graph = graph_std / k_avg_graph
            largest_eigenvalue,largest_eigen_vector = eigsh(nx.adjacency_matrix(G).astype(float), k=1, which='LA', return_eigenvectors=True)
            Beta = float(lam) / largest_eigenvalue[0]
            graph_correlation = nx.degree_assortativity_coefficient(G)
            rho = (np.sum(largest_eigen_vector) / (N * np.sum(largest_eigen_vector ** 3))) * (Beta * largest_eigenvalue[0] - 1)
            parameters = np.array(
                [N, sims, it, k_avg_graph, x, lam, jump, Alpha, Beta, i, tau, Istar, new_trajcetory_bin, dir_path,
                 prog, eps_graph, eps_graph, graph_std, graph_skewness, third_moment, second_moment,graph_correlation,rho])
        np.save('parameters_{}.npy'.format(i), parameters)
        np.save('largest_eigen_vector_{}.npy'.format(i), largest_eigenvalue[0])
        np.save('largest_eigenvalue_{}.npy'.format(i), largest_eigen_vector)
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
            prog_mc = 'gam'
            bank,Num_inital_conditions = 1000000,100
            outfile ='mc_N_{}_eps_{}_R_{}'.format(N,eps_din,lam)
            os.system(dir_path + '/slurm.serjob python3 ' + dir_path + '/gillespierunhomo.py ' + str(prog_mc) + ' ' +
                      str(Alpha) + ' ' + str(bank) + ' ' + str(outfile) + ' ' + str(infile) + ' ' + str(
                Num_inital_conditions) + ' ' + str(Num_inf) + ' ' + str(i) + ' ' + str(Beta))

        # os.system('{} {} {} {}'.format(program_path,path_adj_in,path_adj_out,path_parameters))


if __name__ == '__main__':
    # Parameters for the network
    N = 1800 # number of nodes
    prog = 'gam'
    lam = 1.2 # The reproduction number
    eps_din,eps_dout = 0.6,0.6 # The normalized std (second moment divided by the first) of the network
    correlation = 0.1
    number_of_networks = 10
    k = 20 # Average number of neighbors for each node
    error_graphs = False

    # Parameters for the WE method
    sims = 1000 # Number of simulations at each bin
    tau = 0.5
    it = 70
    jump = 1
    new_trajcetory_bin = 2

    # Parameter that don't get chagne
    relaxation_time  = 20
    x = 0.2 # intial infection percentage
    Num_inf = int(x*N) # Number of initially infected nodes
    Alpha = 1.0 # Recovery rate
    Beta_avg = Alpha * lam / k # Infection rate for each node
    run_mc_simulation = True

    parameters = np.array([N,sims,it,k,x,lam,jump,Num_inf,Alpha,number_of_networks,tau,eps_din,eps_dout,new_trajcetory_bin,prog,Beta_avg,error_graphs,correlation])
    graphname  = 'GNull'
    foldername = 'prog_{}_N{}_k_{}_R_{}_tau_{}_it_{}_jump_{}_new_trajcetory_bin_{}_sims_{}_net_{}_epsin_{}_epsout_{}_correlation_{}_err_{}'.format(
        prog, N, k, lam, tau, it, jump, new_trajcetory_bin, sims, number_of_networks, eps_din, eps_dout,correlation,error_graphs)
    # y1star=(-2*eps_din*(1 + eps_dout*eps_din)+ lam*(-1 + eps_din)*(1 + (-1 + 2*eps_dout)*eps_din)+ np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(-1 +eps_dout)*(-1 +eps_din)*eps_din)
    # y2star=(lam + eps_din*(-2 + 2*lam +lam*eps_din+ 2*eps_dout*(lam +(-1 + lam)*eps_din)) -np.sqrt(lam**2 +eps_din*(4*eps_din +lam**2*eps_din*(-2 +eps_din**2) +4*eps_dout*(lam -(-2 + lam)*eps_din**2) +4*eps_dout**2*eps_din*(lam -(-1 + lam)*eps_din**2))))/(4*lam*(1 +eps_dout)*eps_din*(1 + eps_din))
    # Istar = (y1star +y2star)*N
    Istar = (1 - 1/lam) * N


    # What's the job to run either on the cluster or on the laptop
    job_to_cluster(foldername,parameters,Istar,error_graphs,run_mc_simulation)
    # act_as_main(foldername,parameters,Istar,prog)
