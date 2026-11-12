import tkinter as tk
from tkinter import ttk, messagebox
import time
import heapq
from collections import deque

# ============================================================
# LABORATORIO DE ALGORITMOS DE BUSQUEDA E INTELIGENCIA EVOLUTIVA
# BFS, DFS, Greedy Best-First y A*
# ============================================================

# Ubicaciones del mapa (coordenadas para dibujar el grafo)
NODES = {
    "A": (90, 100),
    "B": (230, 65),
    "C": (370, 100),
    "D": (510, 65),
    "E": (650, 110),
    "F": (110, 260),
    "G": (270, 250),
    "H": (430, 260),
    "I": (590, 260),
    "J": (350, 410),
}

# Conexiones: (nodo1, nodo2, costo)
EDGES = [
    ("A", "B", 4), ("A", "F", 3), ("A", "G", 8),
    ("B", "C", 5), ("B", "G", 2), ("B", "F", 6),
    ("C", "D", 3), ("C", "G", 4), ("C", "H", 7),
    ("D", "E", 6), ("D", "H", 2),
    ("E", "I", 3), ("E", "H", 5),
    ("F", "G", 200), ("F", "J", 100),
    ("G", "H", 3), ("G", "J", 4),
    ("H", "I", 2), ("H", "J", 5),
    ("I", "J", 6),
]

# Conexiones no disponibles. El programa las omite al buscar.
BLOCKED_EDGES = {("C", "H")}

# Heurística: distancia aproximada al destino, basada en las coordenadas.
# Se calcula dinámicamente para el nodo destino seleccionado.
def heuristic(node, goal):
    x1, y1 = NODES[node]
    x2, y2 = NODES[goal]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 / 100


def edge_blocked(a, b):
    return (a, b) in BLOCKED_EDGES or (b, a) in BLOCKED_EDGES


def build_graph():
    graph = {node: [] for node in NODES}
    for a, b, cost in EDGES:
        if not edge_blocked(a, b):
            graph[a].append((b, cost))
            graph[b].append((a, cost))
    return graph


GRAPH = build_graph()


def reconstruct_path(parent, start, goal):
    if goal != start and goal not in parent:
        return []
    path = [goal]
    current = goal
    while current != start:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def path_cost(path):
    total = 0
    for i in range(len(path) - 1):
        for neighbor, cost in GRAPH[path[i]]:
            if neighbor == path[i + 1]:
                total += cost
                break
    return total


def bfs(start, goal):
    queue = deque([start])
    visited = {start}
    parent = {}
    explored = 0

    while queue:
        current = queue.popleft()
        explored += 1
        if current == goal:
            return reconstruct_path(parent, start, goal), explored

        for neighbor, _ in GRAPH[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return [], explored


def dfs(start, goal):
    stack = [start]
    visited = {start}
    parent = {}
    explored = 0

    while stack:
        current = stack.pop()
        explored += 1
        if current == goal:
            return reconstruct_path(parent, start, goal), explored

        # reversed mantiene un orden de exploracion visual consistente
        for neighbor, _ in reversed(GRAPH[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)

    return [], explored


def greedy_best_first(start, goal):
    priority_queue = [(heuristic(start, goal), start)]
    visited = {start}
    parent = {}
    explored = 0

    while priority_queue:
        _, current = heapq.heappop(priority_queue)
        explored += 1
        if current == goal:
            return reconstruct_path(parent, start, goal), explored

        for neighbor, _ in GRAPH[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                heapq.heappush(priority_queue, (heuristic(neighbor, goal), neighbor))

    return [], explored


def astar(start, goal):
    priority_queue = [(heuristic(start, goal), 0, start)]
    g_cost = {start: 0}
    parent = {}
    explored = 0

    while priority_queue:
        _, current_cost, current = heapq.heappop(priority_queue)

        # Ignora entradas antiguas de la cola
        if current_cost != g_cost.get(current, float("inf")):
            continue

        explored += 1
        if current == goal:
            return reconstruct_path(parent, start, goal), explored

        for neighbor, edge_cost in GRAPH[current]:
            new_cost = current_cost + edge_cost
            if new_cost < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = new_cost
                parent[neighbor] = current
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(priority_queue, (priority, new_cost, neighbor))

    return [], explored


ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "Greedy Best-First": greedy_best_first,
    "A*": astar,
}


class RouteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Búsqueda de Rutas - IA")
        self.root.geometry("1000x720")
        self.root.minsize(900, 650)

        self.last_results = {}
        self.create_widgets()
        self.draw_graph()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="LABORATORIO DE ALGORITMOS DE BÚSQUEDA",
            font=("Arial", 18, "bold"),
            fg="#173B65"
        )
        title.pack(pady=(12, 4))

        subtitle = tk.Label(
            self.root,
            text="Sistema de rutas logísticas | BFS • DFS • Greedy Best-First • A*",
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 10))

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")

        ttk.Label(controls, text="Algoritmo:").grid(row=0, column=0, padx=5, pady=5)
        self.algorithm_var = tk.StringVar(value="A*")
        self.algorithm_combo = ttk.Combobox(
            controls, textvariable=self.algorithm_var,
            values=list(ALGORITHMS.keys()), state="readonly", width=22
        )
        self.algorithm_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(controls, text="Inicio:").grid(row=0, column=2, padx=5, pady=5)
        self.start_var = tk.StringVar(value="A")
        self.start_combo = ttk.Combobox(
            controls, textvariable=self.start_var,
            values=list(NODES.keys()), state="readonly", width=5
        )
        self.start_combo.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(controls, text="Destino:").grid(row=0, column=4, padx=5, pady=5)
        self.goal_var = tk.StringVar(value="J")
        self.goal_combo = ttk.Combobox(
            controls, textvariable=self.goal_var,
            values=list(NODES.keys()), state="readonly", width=5
        )
        self.goal_combo.grid(row=0, column=5, padx=5, pady=5)

        ttk.Button(controls, text="Ejecutar búsqueda", command=self.run_search).grid(
            row=0, column=6, padx=8, pady=5
        )
        ttk.Button(controls, text="Comparar todos", command=self.compare_all).grid(
            row=0, column=7, padx=5, pady=5
        )

        self.canvas = tk.Canvas(
            self.root, width=800, height=460, bg="#F7F9FC",
            highlightthickness=1, highlightbackground="#B8C5D6"
        )
        self.canvas.pack(padx=12, pady=8, fill="both", expand=True)

        results_frame = ttk.LabelFrame(self.root, text="Resultados de ejecución", padding=8)
        results_frame.pack(fill="x", padx=12, pady=(0, 10))

        self.result_text = tk.Text(results_frame, height=7, font=("Consolas", 10), wrap="word")
        self.result_text.pack(fill="x")
        self.result_text.insert("1.0", "Selecciona el algoritmo, el nodo inicial y el destino; después pulsa Ejecutar búsqueda.")
        self.result_text.config(state="disabled")

    def draw_graph(self, path=None, start=None, goal=None):
        self.canvas.delete("all")
        path = path or []
        path_edges = set()
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i + 1]))
            path_edges.add((path[i + 1], path[i]))

        # Dibujar conexiones primero
        for a, b, cost in EDGES:
            x1, y1 = NODES[a]
            x2, y2 = NODES[b]
            blocked = edge_blocked(a, b)
            selected = (a, b) in path_edges

            if selected:
                color, width = "#E63946", 4
            elif blocked:
                color, width = "#9A9A9A", 2
            else:
                color, width = "#8CA0B3", 2

            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            label = f"{cost}"
            if blocked:
                label += " X"
            self.canvas.create_rectangle(mx - 15, my - 12, mx + 15, my + 12, fill="#FFFFFF", outline="")
            self.canvas.create_text(
                mx, my, text=label, fill="#B22222" if blocked else "#263746",
                font=("Arial", 9, "bold")
            )

        # Dibujar nodos
        for node, (x, y) in NODES.items():
            if node == start:
                fill = "#2A9D8F"
            elif node == goal:
                fill = "#F4A261"
            elif node in path:
                fill = "#E63946"
            else:
                fill = "#DCE8F5"

            self.canvas.create_oval(
                x - 22, y - 22, x + 22, y + 22,
                fill=fill, outline="#173B65", width=2
            )
            self.canvas.create_text(x, y, text=node, font=("Arial", 12, "bold"), fill="#142536")

        self.canvas.create_text(
            15, 15, anchor="nw",
            text="Leyenda: Inicio = verde | Destino = naranja | Ruta = rojo | Conexión bloqueada = X",
            font=("Arial", 9), fill="#263746"
        )

    def display_results(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def run_search(self):
        algorithm = self.algorithm_var.get()
        start = self.start_var.get()
        goal = self.goal_var.get()

        if start == goal:
            messagebox.showwarning("Selección inválida", "El nodo inicial y el destino deben ser diferentes.")
            return

        function = ALGORITHMS[algorithm]
        begin = time.perf_counter()
        path, explored = function(start, goal)
        elapsed = (time.perf_counter() - begin) * 1000
        cost = path_cost(path) if path else 0

        explanation = {
            "BFS": "Explora por niveles y busca una ruta con el menor número de conexiones.",
            "DFS": "Explora una rama en profundidad antes de retroceder.",
            "Greedy Best-First": "Prioriza el nodo que parece más cercano al destino según la heurística.",
            "A*": "Combina el costo acumulado y la heurística para buscar una ruta de menor costo."
        }[algorithm]

        route_text = " -> ".join(path) if path else "No se encontró una ruta."
        output = (
            f"Algoritmo: {algorithm}\n"
            f"Ruta encontrada: {route_text}\n"
            f"Costo total: {cost}\n"
            f"Nodos explorados: {explored}\n"
            f"Tiempo de ejecución: {elapsed:.4f} ms\n"
            f"Comportamiento: {explanation}"
        )

        self.last_results[algorithm] = {
            "path": route_text, "cost": cost, "explored": explored, "time": elapsed
        }
        self.display_results(output)
        self.draw_graph(path, start, goal)

    def compare_all(self):
        start = self.start_var.get()
        goal = self.goal_var.get()

        if start == goal:
            messagebox.showwarning("Selección inválida", "El nodo inicial y el destino deben ser diferentes.")
            return

        lines = [f"COMPARACIÓN DE ALGORITMOS (Inicio: {start} | Destino: {goal})", "=" * 68]
        best_cost = None
        best_algorithm = None

        for name, function in ALGORITHMS.items():
            begin = time.perf_counter()
            path, explored = function(start, goal)
            elapsed = (time.perf_counter() - begin) * 1000
            cost = path_cost(path) if path else 0
            route_text = " -> ".join(path) if path else "Sin ruta"

            self.last_results[name] = {
                "path": route_text, "cost": cost, "explored": explored, "time": elapsed
            }

            lines.append(
                f"\n{name}\n"
                f"  Ruta: {route_text}\n"
                f"  Costo: {cost} | Explorados: {explored} | Tiempo: {elapsed:.4f} ms"
            )

            if path and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_algorithm = name

        lines.append("\n" + "=" * 68)
        lines.append(f"Menor costo observado: {best_cost} (algoritmo: {best_algorithm})")
        lines.append("Nota: BFS optimiza número de conexiones, no necesariamente el costo.")
        lines.append("Los tiempos y nodos explorados se obtienen al ejecutar el programa.")

        self.display_results("\n".join(lines))
        # Dibuja la ruta del algoritmo A* en el mapa tras comparar
        path, _ = astar(start, goal)
        self.draw_graph(path, start, goal)


if __name__ == "__main__":
    root = tk.Tk()
    app = RouteApp(root)
    root.mainloop()