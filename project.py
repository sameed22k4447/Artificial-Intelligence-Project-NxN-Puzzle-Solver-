from datetime import datetime
from collections import deque
import numpy as np
import heapq
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
class Node:
    def __init__(self, state, g, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g

    def __lt__(self, other):
        return False 

    def equal(self, state):
        return np.array_equal(self.state, state)

    def manhattan_distance(self, goal):
        distance = 0
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                val = self.state[i][j]
                if val != 0:
                    goal_pos = np.argwhere(goal == val)
                    if goal_pos.size > 0:
                        goal_i, goal_j = goal_pos[0]
                        distance += abs(goal_i - i) + abs(goal_j - j)
        return distance

    def expand(self):
        x, y = np.argwhere(self.state == 0)[0]
        directions = {
            "Up": (x - 1, y),
            "Down": (x + 1, y),
            "Left": (x, y - 1),
            "Right": (x, y + 1)
        }
        children = []
        for action, (new_x, new_y) in directions.items():
            if 0 <= new_x < len(self.state) and 0 <= new_y < len(self.state):
                new_state = self.state.copy()
                new_state[x, y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[x, y]
                children.append(Node(new_state, self.g + 1, self, action))
        return children

class GoalTree:
    def __init__(self, initial_state, goal_state):
        self.root = Node(np.array(initial_state), 0)
        self.goal = np.array(goal_state)

    def is_goal(self, state):
        return np.array_equal(state, self.goal)

    def solve(self, strategy):
        start_time = datetime.now()
        if strategy.lower() == 'breadth first':
            result = self.breadth_first()
        elif strategy.lower() == 'depth first':
            result = self.depth_first()
        elif strategy.lower() == 'uniform cost':
            result = self.uniform_cost()
        elif strategy.lower() == 'iterative deepening':
            result = self.iterative_deepening()
        elif strategy.lower() == 'depth limited':
            limit = simpledialog.askinteger("Depth Limit", "Enter depth limit:", parent=self.root)
            if limit is None:
                return None
            result = self.depth_limited(limit)
        elif strategy.lower() == 'a*':
            result = self.a_star()
        elif strategy.lower() == 'greedy best first':
            result = self.greedy_best_first()
        else:
            raise ValueError("Unsupported strategy.")
        end_time = datetime.now()
        return result + (start_time, end_time)

    def iterative_deepening(self):
        depth = 0
        while True:
            result = self.depth_limited(depth)
            if result[5]:
                return result
            depth += 1

    def depth_limited(self, limit):
        frontier = [(self.root, 0)]
        explored = set()
        processed_nodes = 0
        max_stored_nodes = 1

        while frontier:
            max_stored_nodes = max(max_stored_nodes, len(frontier))
            node, depth = frontier.pop()
            processed_nodes += 1

            if self.is_goal(node.state):
                return node.state, self.build_solution(node), node.g, processed_nodes, max_stored_nodes, True

            if depth < limit:
                explored.add(self.hash_state(node.state))
                for child in node.expand():
                    if self.hash_state(child.state) not in explored:
                        frontier.append((child, depth + 1))

        return self.root.state, [], 0, processed_nodes, max_stored_nodes, False

    def breadth_first(self):
        frontier = deque([self.root])
        explored = set()
        processed_nodes = 0
        max_stored_nodes = 1

        while frontier:
            max_stored_nodes = max(max_stored_nodes, len(frontier))
            node = frontier.popleft()
            processed_nodes += 1

            if self.is_goal(node.state):
                return node.state, self.build_solution(node), node.g, processed_nodes, max_stored_nodes, True

            explored.add(self.hash_state(node.state))
            for child in node.expand():
                if self.hash_state(child.state) not in explored:
                    frontier.append(child)

        return self.root.state, [], 0, processed_nodes, max_stored_nodes, False

    def depth_first(self):
        frontier = [self.root]
        explored = set()
        processed_nodes = 0
        max_stored_nodes = 1

        while frontier:
            max_stored_nodes = max(max_stored_nodes, len(frontier))
            node = frontier.pop()
            processed_nodes += 1

            if self.is_goal(node.state):
                return node.state, self.build_solution(node), node.g, processed_nodes, max_stored_nodes, True

            explored.add(self.hash_state(node.state))
            for child in reversed(node.expand()):
                if self.hash_state(child.state) not in explored:
                    frontier.append(child)

        return self.root.state, [], 0, processed_nodes, max_stored_nodes, False

    def greedy_best_first(self):
        frontier = []
        heapq.heappush(frontier, (self.root.manhattan_distance(self.goal), self.root))
        explored = set()
        processed_nodes = 0
        max_stored_nodes = 1

        while frontier:
            max_stored_nodes = max(max_stored_nodes, len(frontier))
            _, node = heapq.heappop(frontier)
            processed_nodes += 1

            if self.is_goal(node.state):
                return node.state, self.build_solution(node), node.g, processed_nodes, max_stored_nodes, True

            explored.add(self.hash_state(node.state))
            for child in node.expand():
                if self.hash_state(child.state) not in explored:
                    h = child.manhattan_distance(self.goal)
                    heapq.heappush(frontier, (h, child))

        return self.root.state, [], 0, processed_nodes, max_stored_nodes, False

    def a_star(self):
        frontier = []
        heapq.heappush(frontier, (self.root.manhattan_distance(self.goal) + self.root.g, self.root))
        explored = set()
        processed_nodes = 0
        max_stored_nodes = 1

        while frontier:
            max_stored_nodes = max(max_stored_nodes, len(frontier))
            _, node = heapq.heappop(frontier)
            processed_nodes += 1

            if self.is_goal(node.state):
                return node.state, self.build_solution(node), node.g, processed_nodes, max_stored_nodes, True

            explored.add(self.hash_state(node.state))
            for child in node.expand():
                if self.hash_state(child.state) not in explored:
                    f = child.g + child.manhattan_distance(self.goal)
                    heapq.heappush(frontier, (f, child))

        return self.root.state, [], 0, processed_nodes, max_stored_nodes, False

    def uniform_cost(self):
        frontier = []
        heapq.heappush(frontier, (self.root.g, self.root))
        explored = set()
        processed_nodes = 0
        max_stored_nodes = 1

        while frontier:
            max_stored_nodes = max(max_stored_nodes, len(frontier))
            cost, node = heapq.heappop(frontier)
            processed_nodes += 1

            if self.is_goal(node.state):
                return node.state, self.build_solution(node), node.g, processed_nodes, max_stored_nodes, True

            explored.add(self.hash_state(node.state))
            for child in node.expand():
                if self.hash_state(child.state) not in explored:
                    heapq.heappush(frontier, (child.g, child))

        return self.root.state, [], 0, processed_nodes, max_stored_nodes, False

    def build_solution(self, node):
        actions = []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]

    def hash_state(self, state):
        return tuple(state.flatten())

class NPuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("N Puzzle Solver")
        self.root.geometry("1000x800")
        
        self.dim = tk.IntVar(value=3)
        self.algorithm = tk.StringVar(value="A*")
        self.start_state = []
        self.goal_state = []
        self.solution_moves = []
        self.current_move = 0
        self.start_labels = []
        self.goal_labels = []
        self.solution_labels = []
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        input_frame = ttk.LabelFrame(main_frame, text="Puzzle Configuration", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Puzzle Dimension:").grid(row=0, column=0, sticky=tk.W)
        self.dim_combobox = ttk.Combobox(input_frame, textvariable=self.dim, 
                                       values=[2, 3, 4], state="readonly")
        self.dim_combobox.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.dim_combobox.bind("<<ComboboxSelected>>", self.update_dimension)
        
        ttk.Button(input_frame, text="Set Start State", command=self.set_start_state).grid(row=1, column=0, pady=5)
        
        ttk.Button(input_frame, text="Set Goal State", command=self.set_goal_state).grid(row=1, column=1, pady=5)
        
        ttk.Label(input_frame, text="Algorithm:").grid(row=2, column=0, sticky=tk.W)
        self.algo_combobox = ttk.Combobox(input_frame, textvariable=self.algorithm, 
                                         values=["Breadth First", "Depth First", "Uniform Cost", 
                                                 "Iterative Deepening", "A*", 
                                                "Greedy Best First"], state="readonly")
        self.algo_combobox.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(input_frame, text="Solve Puzzle", command=self.solve_puzzle).grid(row=3, column=0, columnspan=2, pady=10)
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.start_frame = ttk.LabelFrame(display_frame, text="Start State", padding="10")
        self.start_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.goal_frame = ttk.LabelFrame(display_frame, text="Goal State", padding="10")
        self.goal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.solution_frame = ttk.LabelFrame(display_frame, text="Solution State", padding="10")
        self.solution_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.X, pady=5)
        
        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_move).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_move).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Reset", command=self.reset_view).pack(side=tk.RIGHT, padx=5)
        self.init_puzzle_displays()
        
    def update_dimension(self, event=None):
        self.init_puzzle_displays()
        self.start_state = []
        self.goal_state = []
        self.solution_moves = []
        self.current_move = 0
        self.update_puzzle_display(self.start_labels, [])
        self.update_puzzle_display(self.goal_labels, [])
        self.update_puzzle_display(self.solution_labels, [])
        self.results_text.delete(1.0, tk.END)
        
    def init_puzzle_displays(self):
        dim = self.dim.get()
        for frame in [self.start_frame, self.goal_frame, self.solution_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        self.start_labels = self.create_grid(self.start_frame, dim)
        self.goal_labels = self.create_grid(self.goal_frame, dim)
        self.solution_labels = self.create_grid(self.solution_frame, dim)
        
    def create_grid(self, parent, dim):
        labels = []
        for i in range(dim):
            row = []
            for j in range(dim):
                label = ttk.Label(parent, text="", width=4, relief="solid", 
                                anchor="center", font=('Helvetica', 12))
                label.grid(row=i, column=j, padx=2, pady=2, 
                          sticky="nsew", ipadx=5, ipady=5)
                
                parent.grid_rowconfigure(i, weight=1)
                parent.grid_columnconfigure(j, weight=1)
                row.append(label)
            labels.append(row)
        return labels
        
    def set_start_state(self):
        dim = self.dim.get()
        self.start_state = self.get_state_from_user("Start State", dim)
        if self.start_state:
            self.update_puzzle_display(self.start_labels, self.start_state)
        
    def set_goal_state(self):
        dim = self.dim.get()
        self.goal_state = self.get_state_from_user("Goal State", dim)
        if self.goal_state:
            self.update_puzzle_display(self.goal_labels, self.goal_state)
        
    def get_state_from_user(self, title, dim):
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry(f"{min(400, 100*dim)}x{min(400, 100*dim)}")
        
        entries = []
        for i in range(dim):
            row = []
            for j in range(dim):
                entry = ttk.Entry(top, width=5, font=('Helvetica', 12))
                entry.grid(row=i, column=j, padx=5, pady=5)
                row.append(entry)
            entries.append(row)
            
        def on_ok():
            try:
                state = []
                for i in range(dim):
                    row = []
                    for j in range(dim):
                        val = int(entries[i][j].get())
                        row.append(val)
                    state.append(row)
                    
                flat = [num for row in state for num in row]
                if sorted(flat) != list(range(dim*dim)):
                    raise ValueError(f"State must contain all numbers from 0 to {dim*dim-1} with no duplicates")
                    
                top.state = state
                top.destroy()
            except ValueError as e:
                messagebox.showerror("Error", str(e), parent=top)
                
        ttk.Button(top, text="OK", command=on_ok).grid(row=dim, column=0, columnspan=dim, pady=10)
        
        top.wait_window()
        return getattr(top, 'state', None)
    
    def update_puzzle_display(self, labels, state):
        if not state:
            for row in labels:
                for label in row:
                    label.config(text="")
            return
            
        dim = len(state)
        for i in range(dim):
            for j in range(dim):
                val = state[i][j]
                labels[i][j].config(text=str(val) if val != 0 else "")
                
    def solve_puzzle(self):
        if not self.start_state or not self.goal_state:
            messagebox.showerror("Error", "Please set both start and goal states first")
            return
            
        algorithm_map = {
            "Breadth First": "breadth first",
            "Depth First": "depth first",
            "Uniform Cost": "uniform cost",
            "Depth Limited": "depth limited",
            "Iterative Deepening": "iterative deepening",
            "A*": "a*",
            "Greedy Best First": "greedy best first"
        }
        
        algorithm = algorithm_map.get(self.algorithm.get(), "a*")
        
        gt = GoalTree(self.start_state, self.goal_state)
        result = gt.solve(algorithm)
        
        if result is None:  
            return
            
        sol_state, moves, g, processed_nodes, max_stored_nodes, success, start_time, end_time = result
            
        self.solution_moves = moves
        self.current_move = 0
    
        self.update_puzzle_display(self.solution_labels, self.start_state)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=== Output Information ===\n")
        self.results_text.insert(tk.END, f"Time taken: {end_time - start_time}\n")
        self.results_text.insert(tk.END, f"G-value (depth/level found): {g}\n")
        self.results_text.insert(tk.END, f"Nodes processed: {processed_nodes}\n")
        self.results_text.insert(tk.END, f"Max nodes stored: {max_stored_nodes}\n")
        self.results_text.insert(tk.END, f"Solution found?: {success}\n")
        
        if success:
            self.results_text.insert(tk.END, "\nMoves to solve:\n")
            self.results_text.insert(tk.END, ' -> '.join(moves) if moves else "Already solved!")

            self.highlight_solution_path()
        else:
            self.results_text.insert(tk.END, "\nNo solution found.")
            
    def highlight_solution_path(self):
        current_state = [row[:] for row in self.start_state]
        dim = len(current_state)
        
        blank_pos = None
        for i in range(dim):
            for j in range(dim):
                if current_state[i][j] == 0:
                    blank_pos = (i, j)
                    break
            if blank_pos:
                break
            
        for move in self.solution_moves:
            i, j = blank_pos
            if move == "Up":
                current_state[i][j], current_state[i-1][j] = current_state[i-1][j], current_state[i][j]
                blank_pos = (i-1, j)
            elif move == "Down":
                current_state[i][j], current_state[i+1][j] = current_state[i+1][j], current_state[i][j]
                blank_pos = (i+1, j)
            elif move == "Left":
                current_state[i][j], current_state[i][j-1] = current_state[i][j-1], current_state[i][j]
                blank_pos = (i, j-1)
            elif move == "Right":
                current_state[i][j], current_state[i][j+1] = current_state[i][j+1], current_state[i][j]
                blank_pos = (i, j+1)
                
        self.update_puzzle_display(self.solution_labels, current_state)
        
    def prev_move(self):
        if not self.solution_moves or self.current_move <= 0:
            return
            
        self.current_move -= 1
        self.show_move(self.current_move)
        
    def next_move(self):
        if not self.solution_moves or self.current_move >= len(self.solution_moves):
            return
            
        self.current_move += 1
        self.show_move(self.current_move - 1)  
        
    def show_move(self, move_index):
        current_state = [row[:] for row in self.start_state]
        dim = len(current_state)
        
        blank_pos = None
        for i in range(dim):
            for j in range(dim):
                if current_state[i][j] == 0:
                    blank_pos = (i, j)
                    break
            if blank_pos:
                break
                
        for i in range(move_index + 1):
            move = self.solution_moves[i]
            x, y = blank_pos
            
            if move == "Up":
                current_state[x][y], current_state[x-1][y] = current_state[x-1][y], current_state[x][y]
                blank_pos = (x-1, y)
            elif move == "Down":
                current_state[x][y], current_state[x+1][y] = current_state[x+1][y], current_state[x][y]
                blank_pos = (x+1, y)
            elif move == "Left":
                current_state[x][y], current_state[x][y-1] = current_state[x][y-1], current_state[x][y]
                blank_pos = (x, y-1)
            elif move == "Right":
                current_state[x][y], current_state[x][y+1] = current_state[x][y+1], current_state[x][y]
                blank_pos = (x, y+1)
                
        self.update_puzzle_display(self.solution_labels, current_state)
        
        self.results_text.tag_remove("highlight", 1.0, tk.END)
        if move_index >= 0 and move_index < len(self.solution_moves):
            start_idx = self.results_text.search(' -> '.join(self.solution_moves), 1.0, tk.END)
            if start_idx:
                start_idx = self.results_text.index(start_idx)
                end_idx = self.results_text.index(f"{start_idx}+{len(' -> '.join(self.solution_moves))}c")
                self.results_text.tag_add("highlight", start_idx, end_idx)
                self.results_text.tag_config("highlight", background="yellow")
                
    def reset_view(self):
        self.current_move = 0
        self.update_puzzle_display(self.solution_labels, self.start_state)
        self.results_text.tag_remove("highlight", 1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = NPuzzleGUI(root)
    root.mainloop()