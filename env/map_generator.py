import random

def generate_map(grid=8, density=0.15, seed=None):
    """Generate a random grid map with start, goal, and holes.

    Args:
        grid (int, optional): The size of the square grid. Defaults to 8.
        density (float, optional): The density of holes in the grid. Defaults to 0.15.
        seed (int, optional): The random seed for reproducibility. Defaults to None.

    Returns:
        list[str]: A list of strings representing the grid map.
    """

    if seed is not None:
        random.seed(seed)

    while True:
        cells = ["F"] * (grid * grid)
        
        cells[0] = "S"
        cells[-1] = "G"
        
        num_holes = int(grid * grid * density)
        max_holes = (grid * grid) - 2
        if num_holes > max_holes:
            num_holes = max_holes
            
        hole_positions = random.sample(range(1, grid * grid - 1), num_holes)
        for p in hole_positions:
            cells[p] = "H"
        
        grid_map = ["".join(cells[r * grid:(r + 1) * grid]) for r in range(grid)]
        
        visited = set()
        queue = [(0, 0)]
        path_found = False
        
        while queue:
            r, c = queue.pop(0)
            
            if (r, c) == (grid - 1, grid - 1):
                path_found = True
                break
            
            if (r, c) in visited:
                continue
            
            visited.add((r, c))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < grid and 0 <= nc < grid and grid_map[nr][nc] != "H":
                    if (nr, nc) not in visited:
                        queue.append((nr, nc))
        
        if path_found:
            return grid_map