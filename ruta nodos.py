import heapq
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ======================
# Datos del problema
# ======================
romania_map = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Dobreta': 75},
    'Dobreta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Dobreta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

heuristics = {
    'Arad': 366, 'Zerind': 374, 'Oradea': 380, 'Sibiu': 253,
    'Timisoara': 329, 'Lugoj': 0, 'Mehadia': 241, 'Dobreta': 242,
    'Craiova': 160, 'Rimnicu Vilcea': 193, 'Fagaras': 176, 'Pitesti': 100,
    'Bucharest': 0, 'Giurgiu': 77, 'Urziceni': 80, 'Hirsova': 151,
    'Eforie': 161, 'Vaslui': 199, 'Iasi': 226, 'Neamt': 234
}

# ======================
# Clase Nodo
# ======================
class Node:
    def __init__(self, city, parent=None, g=0, h=0):
        self.city = city
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

    def path(self):
        node, p = self, []
        while node:
            p.append(node)
            node = node.parent
        return p[::-1]

# ======================
# Algoritmo A*
# ======================
def a_star_limited(start, goal):
    open_list = []
    open_dict = {}
    closed_set = set()
    explored_nodes = []
    optimal_path = None
    best_f = float('inf')
    start_node = Node(start, None, 0, heuristics[start])
    heapq.heappush(open_list, start_node)
    open_dict[start] = start_node

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.city not in open_dict or open_dict[current_node.city].f < current_node.f:
            continue
        explored_nodes.append(current_node)
        del open_dict[current_node.city]

        if current_node.city == goal and current_node.f < best_f:
            optimal_path = current_node.path()
            best_f = current_node.f
            continue
        if optimal_path and current_node.f >= best_f:
            continue
        closed_set.add(current_node.city)

        for neighbor, distance in romania_map[current_node.city].items():
            if neighbor in closed_set:
                continue

            g = current_node.g + distance
            h = heuristics[neighbor]
            child = Node(neighbor, current_node, g, h)

            if neighbor not in open_dict or child.f < open_dict[neighbor].f:
                heapq.heappush(open_list, child)
                open_dict[neighbor] = child

    return optimal_path, explored_nodes

# ======================
# Visualización Vertical
# ======================
def plot_vertical_tree(explored_nodes, optimal_path=None):
    plt.figure(figsize=(14, 10))
    G = nx.DiGraph()
    pos = {}
    node_positions = {}
    subtree_widths = {}
    
    # Calcular anchos de subárboles
    def calculate_subtree_width(node):
        if node.city in subtree_widths:
            return subtree_widths[node.city]
        
        if not any(n.parent == node for n in explored_nodes):
            subtree_widths[node.city] = 1
            return 1
        
        width = 0
        for child in [n for n in explored_nodes if n.parent == node]:
            width += calculate_subtree_width(child)
        
        subtree_widths[node.city] = max(1, width)
        return subtree_widths[node.city]
    
    for node in explored_nodes:
        calculate_subtree_width(node)
    
    # Organizar por niveles
    levels = {}
    for node in explored_nodes:
        depth = len(node.path()) - 1
        if depth not in levels:
            levels[depth] = []
        levels[depth].append(node)
    
    vertical_spacing = 1.8
    horizontal_spacing = 1.5
    
    # Asignar posiciones verticales (de arriba hacia abajo)
    for depth in sorted(levels.keys()):
        y = -depth * vertical_spacing  # Nodo raíz en y=0, hijos en y=-1.8, etc.
        
        # Ordenar nodos por posición de padre
        if depth == 0:
            nodes = levels[depth]
        else:
            nodes = sorted(levels[depth], key=lambda n: node_positions[n.parent.city][0] if n.parent else 0)
        
        current_x = -sum(subtree_widths[n.city] for n in nodes) * horizontal_spacing / 2
        
        for node in nodes:
            if depth == 0:
                x = 0  # Nodo raíz centrado
            else:
                parent_pos = node_positions[node.parent.city]
                siblings = [n for n in levels[depth] if n.parent == node.parent]
                sibling_index = siblings.index(node)
                
                # Calcular posición basada en el ancho del subárbol
                total_width = sum(subtree_widths[n.city] for n in siblings) * horizontal_spacing
                offset = -total_width / 2
                
                for i, sibling in enumerate(siblings[:sibling_index]):
                    offset += subtree_widths[sibling.city] * horizontal_spacing
                
                x = parent_pos[0] + offset + (subtree_widths[node.city] * horizontal_spacing) / 2
            
            node_label = f"{node.city}\n{node.g}+{node.h}={node.f}"
            pos[node_label] = (x, y)
            node_positions[node.city] = (x, y)
    
    # Crear nodos y aristas
    for node in explored_nodes:
        node_label = f"{node.city}\n{node.g}+{node.h}={node.f}"
        G.add_node(node_label)
        if node.parent:
            parent_label = f"{node.parent.city}\n{node.parent.g}+{node.parent.h}={node.parent.f}"
            G.add_edge(parent_label, node_label)
    
    # Colores y estilos
    node_colors = []
    edge_colors = []
    optimal_nodes = set()
    continued_nodes = set()

    if optimal_path:
        for node in optimal_path:
            optimal_nodes.add(f"{node.city}\n{node.g}+{node.h}={node.f}")
        for node in explored_nodes:
            label = f"{node.city}\n{node.g}+{node.h}={node.f}"
            for child in explored_nodes:
                if child.parent == node:
                    continued_nodes.add(label)
                    break

    for node in G.nodes():
        if node in optimal_nodes:
            node_colors.append('#90EE90')  # Verde claro
            edge_colors.append('green')
        elif node in continued_nodes:
            node_colors.append('#ADD8E6')  # Azul claro
            edge_colors.append('blue')
        else:
            node_colors.append('#F5F5F5')   # Gris claro
            edge_colors.append('gray')

    # Dibujar el grafo
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2800,
        node_color=node_colors,
        edge_color=edge_colors,
        font_size=8,
        font_weight='bold',
        arrowsize=18,
        width=1.3,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1'
    )

    # Añadir numeración
    for i, (node, (x, y)) in enumerate(pos.items()):
        plt.text(x, y + 0.45, str(i+1), ha='center', va='center', 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    # Leyenda
    legend_elements = [
        Patch(facecolor='#90EE90', edgecolor='green', label='Ruta óptima'),
        Patch(facecolor='#ADD8E6', edgecolor='blue', label='Rutas expandidas'),
        Patch(facecolor='#F5F5F5', edgecolor='gray', label='No expandidas')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    plt.title("Árbol de Búsqueda A* (Orientación Vertical)", size=14, pad=20)
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

# ======================
# Programa principal
# ======================
def main():
    start, goal = 'Hirsova', 'Lugoj'  # nodo de inicio y nodo final a cambiar segun sea el caso 

    if start not in romania_map or goal not in romania_map:
        print("Error: Nodo no encontrado en el mapa.")
        return

    print(f"\nBuscando desde {start} hasta {goal}...")
    
    optimal_path, explored_nodes = a_star_limited(start, goal)

    if optimal_path:
        print("\nRuta óptima encontrada:")
        for node in optimal_path:
            print(f"{node.city}: {node.g} + {node.h} = {node.f}")
        print(f"\nResumen:")
        print(f"- Nodos explorados: {len(explored_nodes)}")
        print(f"- Pasos en ruta: {len(optimal_path)-1}")
        print(f"- Costo total: {optimal_path[-1].f}")
    else:
        print("No se encontró ruta óptima.")

    plot_vertical_tree(explored_nodes, optimal_path)

if __name__ == "__main__":
    main()
