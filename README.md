# Artificial-Intelligence-Project-NxN-Puzzle-Solver-
A N by N puzzle solver using various search algorithms.
# N x N Puzzle Solver – AI Search Algorithms

This project is an intelligent puzzle-solving application that uses classic AI search algorithms to solve N x N sliding tile puzzles (like the 8-puzzle and 15-puzzle). It features a graphical interface and supports multiple informed and uninformed search strategies.

---

# Features

- Supports puzzles of size **2x2**, **3x3**, and **4x4**
- Visual, interactive **Tkinter-based GUI**
- Allows user input for **custom start and goal states**
- Step-by-step **solution navigation**
- Displays **performance metrics** (time, nodes processed, memory used)
- Implements the following search algorithms:
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
  - Uniform-Cost Search (UCS)
  - Iterative Deepening Search (IDS)
  - A* Search (with Manhattan Distance)
  - Greedy Best-First Search

---

# Algorithms Overview

Each algorithm explores the puzzle state space differently:

- **BFS**: Explores all nodes level by level; guarantees shortest solution in terms of moves.
- **DFS**: Dives deep into one path before backtracking; memory-efficient but not always optimal.
- **UCS**: Explores cheapest path by total cost; optimal when all costs are non-negative.
- **Greedy Best-First**: Uses only heuristic to find the goal quickly; not guaranteed to be optimal.
- **Iterative Deepening**: Repeated DFS with increasing depth limits; combines low memory with completeness.
- **A\***: Combines path cost and heuristic; efficient and optimal with an admissible heuristic.

---

# GUI Demonstration

The application provides:
- **Start State**, **Goal State**, and **Solution View** side by side
- Buttons to **input puzzle**, **choose algorithm**, and **solve**
- Solution path view with **“Next”**, **“Previous”**, and **“Reset”** controls
- Detailed result summary:
  - Time taken
  - Depth of solution
  - Nodes processed
  - Max memory used

---

# How to Run

### 🛠 Prerequisites
- Python 3.x
- Required libraries: `numpy`, `tkinter` (built-in for most Python installations)

# To run the app:
```bash
python project.py
