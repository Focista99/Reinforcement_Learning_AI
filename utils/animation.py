import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np

from env.uboat_env import make_env, make_base_env
from env.wrappers import ObsWrapper, RewWrapper, FuelWrapper
from env.map_generator import generate_map

GRID = 8

COLORS = {
    "S": "#1565C0",  
    "F": "#E3F2FD",  
    "H": "#263238",  
    "G": "#2E7D32"   
}

ICON_STYLE = {
    "H": ("X", "#FF1744", 16), 
    "G": ("*", "#FFD600", 18), 
    "S": ("o", "#FFFFFF", 14)  
}


def draw_map(ax, grid_map):
    """
    Draws the static grid map on the given Matplotlib axes.

    Args:
        ax (matplotlib.axes.Axes): Axes object on which the map is drawn.
        grid_map (list[str]): The grid map to draw (same one used for the episode).
    """
    ax.set(
        xlim=(-0.5, GRID - 0.5),
        ylim=(GRID - 0.5, -0.5),
        aspect="equal",
        facecolor="#90CAF9",
        xticks=range(GRID),
        yticks=range(GRID),
        xticklabels=[],
        yticklabels=[]
    )
    
    for s in ax.spines.values():
        s.set(edgecolor="#0D47A1", linewidth=2)
        
    for r in range(GRID):
        for c in range(GRID):
            cell_char = grid_map[r][c]
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=COLORS.get(cell_char, "#E3F2FD"),
                    edgecolor="#78909C",
                    lw=0.8,
                    zorder=1
                )
            )
            if cell_char in ICON_STYLE:
                sym, col, sz = ICON_STYLE[cell_char]
                ax.text(
                    c, r, sym,
                    ha="center", va="center",
                    fontsize=sz, color=col,
                    fontweight="bold", zorder=2
                )


def get_path(agent, seed=None):
    """
    Runs a single episode with the given agent and records the trajectory.

    Uses make_env() with the given seed so the map is consistent with
    what the agent learned. The wrapper order matches training exactly:
    ObsWrapper -> RewWrapper -> FuelWrapper.

    Args:
        agent: Trained agent exposing an ``act(obs)`` method and an ``eps``
            attribute for the exploration rate.
        seed (int, optional): Map seed. If None, a random seed is used.

    Returns:
        tuple: (path, grid_map) where path is a list of
            (row, col, success, failure) tuples and grid_map is the map
            used in this episode (for drawing).
    """
    # FIX: use make_env() so the wrapper stack matches training exactly
    # (ObsWrapper -> RewWrapper -> FuelWrapper)
    env = make_env(seed=seed)
    # Grab the map that was generated for this episode so we can draw it
    grid_map = generate_map(grid=GRID, density=0.15, seed=seed)

    obs, _ = env.reset()

    original_eps = getattr(agent, 'eps', 0.0)
    agent.eps = 0.0  # greedy during animation — show what the agent learned

    base_env = env.unwrapped
    raw_state = base_env.s
    path = [(raw_state // GRID, raw_state % GRID, False, False)]

    for _ in range(60):
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        raw_state = base_env.s
        success = terminated and reward >= 5.0
        failure = (terminated and not success) or truncated

        path.append((raw_state // GRID, raw_state % GRID, success, failure))

        if terminated or truncated:
            break

    agent.eps = original_eps
    env.close()
    return path, grid_map


def run_animation(agent, n=6, seed=None):
    """
    Generates and displays an animated visualization of ``n`` agent missions.

    All missions use the same seed (and therefore the same map) as training
    so the agent navigates terrain it actually learned.

    Args:
        agent: Trained agent used to generate paths via ``get_path``.
            Must expose ``act(obs)`` and ``eps`` attributes.
        n (int, optional): Number of missions to generate and animate.
            Defaults to 6.
        seed (int, optional): Map seed shared with training. All missions
            reuse this seed so the map is consistent. If None, a single
            random seed is chosen and reused for all missions.

    Returns:
        matplotlib.animation.FuncAnimation: The active animation object.
    """
    if seed is None:
        seed = random.randint(0, 9999)

    print(f"Generating {n} mission paths...")
    # FIX: all missions use the same seed — same map the agent was trained on
    results = [get_path(agent, seed=seed) for _ in range(n)]
    paths = [r[0] for r in results]
    maps  = [r[1] for r in results]  # all identical (same seed), but kept per-mission for clarity

    for i, p in enumerate(paths):
        status = "SUCCESS" if p[-1][2] else "SUNK"
        print(f"  Mission {i+1}: {len(p)} steps -> {status}")

    fig = plt.figure(figsize=(13, 7), facecolor="#0D1B2A")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.05)
    
    ax_map = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    
    ax_info.set(facecolor="#0D1B2A", xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
    for s in ax_info.spines.values():
        s.set_visible(False)
        
    draw_map(ax_map, maps[0])

    marker_sub, = ax_map.plot([], [], "s", ms=22, color="#FF5722", mec="white", mew=2, zorder=5)
    label_u = ax_map.text(0, 0, "U", ha="center", va="center", fontsize=11, color="white", fontweight="bold", zorder=6, visible=False)
    line_trail, = ax_map.plot([], [], "o-", color="#FFA726", ms=4, alpha=0.5, zorder=4, lw=1.5)
    
    text_result = ax_map.text(
        3.5, 3.5, "",
        ha="center", va="center", fontsize=20, fontweight="bold", color="white", zorder=10,
        bbox=dict(facecolor="#0D1B2A", alpha=0, boxstyle="round,pad=0.6"),
        visible=False
    )
    
    title_text = fig.text(0.38, 0.97, "", ha="center", va="top", fontsize=12, color="white", fontweight="bold")
    
    info_texts = [
        ax_info.text(0.05, 0.95 - i * 0.09, "", fontsize=10, color="white", fontfamily="monospace", va="top", transform=ax_info.transAxes)
        for i in range(10)
    ]
    
    state = {
        "mission_idx": 0,
        "step_idx": 0,
        "hold_frames": 0 
    }

    def update(_):
        m_idx = state["mission_idx"]
        s_idx = state["step_idx"]

        path = paths[m_idx]

        if s_idx >= len(path):
            state["hold_frames"] += 1
            if state["hold_frames"] > 12: 
                # --- RESET PARA LA SIGUIENTE MISIÓN ---
                state["mission_idx"] = (m_idx + 1) % n
                state["step_idx"] = 0
                state["hold_frames"] = 0
                
                # 1. Escondemos el cartel de resultado
                text_result.set_visible(False)
                text_result.get_bbox_patch().set_alpha(0)
                
                # 2. Limpiamos solo los datos de la estela y el submarino
                # NO USAMOS ax_map.cla() para no borrar los artistas
                line_trail.set_data([], [])
                marker_sub.set_data([], [])
                label_u.set_visible(False) 

            return [marker_sub, label_u, line_trail, text_result, title_text] + info_texts

        r, c, success, failed = path[s_idx]
        
        marker_sub.set_data([c], [r])
        if success:
            marker_sub.set_color("#4CAF50")
        elif failed:
            marker_sub.set_color("#F44336") 
        else:
            marker_sub.set_color("#FF5722") 
            
        label_u.set_position((c, r))
        label_u.set_visible(True)
        
        trail_x = [path[i][1] for i in range(s_idx + 1)]
        trail_y = [path[i][0] for i in range(s_idx + 1)]
        line_trail.set_data(trail_x, trail_y)
        
        title_text.set_text(f"U-BOOT OPERATION  |  Mission {m_idx + 1}/{n}  |  Step {s_idx + 1}/{len(path)}")
        
        max_steps = max(len(path) - 1, 1)
        fuel_pct = int(100 * (1 - s_idx / max_steps))
        fuel_bar = '#' * (fuel_pct // 10) + '-' * (10 - fuel_pct // 10)
        
        torps_left = max(0, 8 - (s_idx // 3))
        torp_bar = '+' * torps_left + '.' * (8 - torps_left)

        info_data = [
            ("=== U-BOOT STATUS ===", "#90CAF9"),
            (f"Mission : {m_idx + 1}/{n}", "#FFD54F"),
            (f"Step    : {s_idx + 1:3d}", "#FFD54F"),
            (f"Pos     : R{r}  C{c}", "#FFD54F"),
            (f"Fuel    : {fuel_bar}", "#4FC3F7"),
            (f"Torpedoes: {torp_bar}", "#EF9A9A"),
            ("-" * 26, "#546E7A"),
            ("LEGEND:", "#B0BEC5"),
            ("o BASE  X MINE  * CONVOY", "#B0BEC5"),
            ("- Path taken", "#FFA726")
        ]
        
        for txt_widget, (content, color) in zip(info_texts, info_data):
            txt_widget.set_text(content)
            txt_widget.set_color(color)
            
        if success or failed:
            msg = "CONVOY SUNK!" if success else "U-BOOT SUNK!"
            color_msg = "#4CAF50" if success else "#F44336"
            
            text_result.set_text(msg)
            text_result.set_color(color_msg)
            text_result.get_bbox_patch().set_alpha(0.88)
            text_result.set_visible(True)
            
            state["step_idx"] = len(path) 
        else:
            state["step_idx"] += 1
            
        return [marker_sub, label_u, line_trail, text_result, title_text] + info_texts

    ani = animation.FuncAnimation(
        fig, update, interval=320, cache_frame_data=False, repeat=True
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return ani