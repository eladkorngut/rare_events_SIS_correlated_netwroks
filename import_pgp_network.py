import networkx as nx
import pickle

# Function to parse the vertices and edges
def pgp_read(file_path):
    # file_path = './PGPgiantcompo.net'
    # Parsing both vertices and edges sections

    # Create an empty graph
    graph = nx.Graph()


    with open(file_path, 'r') as file:
        reading_vertices = False
        reading_edges = False

        for line in file:
            line = line.strip()

            # Handle section markers
            if line.lower().startswith('*vertices'):
                reading_vertices = True
                reading_edges = False
                continue
            elif line.lower().startswith('*edges') or line.lower().startswith('*arcs'):
                reading_vertices = False
                reading_edges = True
                continue

            # Parse vertices
            if reading_vertices:
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])  # Vertex ID
                    label = parts[1].strip('"') if len(parts) > 1 else None  # Optional label
                    graph.add_node(node_id, label=label)

            # Parse edges
            if reading_edges:
                try:
                    u, v = map(int, line.split()[:2])
                    graph.add_edge(u, v)
                except ValueError:
                    continue
    return graph
    # Display basic graph information
    # nx.info(graph)
if __name__=='__main__':
    G = pgp_read('./PGPgiantcompo.net')
    with open('PGPgiantcompo.nx', 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
