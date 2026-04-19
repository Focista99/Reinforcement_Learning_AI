import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np

from env.uboat_env import make_base_env
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


def draw_map(ax):
    """
    Draws the static grid map on the given Matplotlib axes.

    Generates a fixed map using ``generate_map`` with a constant seed,
    then renders each cell as a colored rectangle and overlays the
    corresponding icon symbol for special cells (mines, goal, start).

    Args:
        ax (matplotlib.axes.Axes): Axes object on which the map is drawn.
            All previous content on the axes is preserved; new patches and
            text artists are added on top.
    """
    static_map = generate_map(grid=GRID, density=0.15, seed=42)
    
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
            cell_char = static_map[r][c]
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


def get_path(agent, idx=0):
    """
    Runs a single episode with the given agent and records the trajectory.

    Creates a fresh environment for each call (random seed), temporarily
    raises the agent's exploration rate to ``0.15`` to introduce slight
    stochasticity, and collects ``(row, col, success, failure)`` tuples
    for every step until the episode ends or 60 steps are reached.

    Args:
        agent: Trained agent exposing an ``act(obs)`` method and an ``eps``
            attribute for the exploration rate.
        idx (int, optional): Unused index kept for call-site consistency
            when generating multiple paths. Defaults to 0.

    Returns:
        list[tuple[int, int, bool, bool]]: Sequence of
            ``(row, col, success, failure)`` tuples representing the
            agent's trajectory. The first element is the starting position;
            ``success`` and ``failure`` are ``True`` only on the terminal step.
    """
    base = make_base_env(seed=random.randint(0, 9999))
    env = ObsWrapper(FuelWrapper(RewWrapper(base)))
    
    obs, _ = env.reset()
    
    original_eps = getattr(agent, 'eps', 0.0)
    agent.eps = 0.15
    
    raw_state = base.unwrapped.s
    path = [(raw_state // GRID, raw_state % GRID, False, False)]
    
    for _ in range(60): 
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        raw_state = base.unwrapped.s
        success = terminated and reward >= 5.0
        failure = terminated and not success
        
        path.append((raw_state // GRID, raw_state % GRID, success, failure))
        
        if terminated or truncated:
            break
            
    agent.eps = original_eps
    env.close()
    return path


def run_animation(agent, n=6):
    """
    Generates and displays an animated visualization of ``n`` agent missions.

    Collects trajectory paths by running the agent through ``n`` independent
    episodes, then renders a looping Matplotlib animation that shows:

    - The submarine marker moving step-by-step along the recorded path.
    - A color-coded trail fading behind the agent.
    - A live status panel (mission index, step count, position, fuel bar,
    torpedo bar, and legend).
    - An overlay message (``"CONVOY SUNK!"`` or ``"U-BOOT SUNK!"``) at
    the end of each mission.

    Missions cycle automatically: after a short hold the map is redrawn
    and the next mission begins.

    Args:
        agent: Trained agent used to generate paths via ``get_path``.
            Must expose ``act(obs)`` and ``eps`` attributes.
        n (int, optional): Number of missions to generate and animate.
            Defaults to 6.

    Returns:
        matplotlib.animation.FuncAnimation: The active animation object.
            Keep a reference to this object to prevent garbage collection
            while the window is open.
    """
    print(f"Generating {n} mission paths...")
    paths = [get_path(agent, i) for i in range(n)]
    
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
        
    draw_map(ax_map)

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
        """
        Frame update callback for the FuncAnimation loop.

        Advances the current mission by one step per call. When the path is
        exhausted, a hold counter delays the transition before resetting to
        the next mission and redrawing the map. On terminal steps, the result
        overlay is shown and the step index is pinned to the end of the path
        so the overlay persists during the hold period.

        Args:
            _ : Frame index supplied by FuncAnimation (unused).

        Returns:
            list: Matplotlib artists that were modified, enabling blitting
                if supported by the backend.
        """
        m_idx = state["mission_idx"]
        s_idx = state["step_idx"]
        path = paths[m_idx]
        
        if s_idx >= len(path):
            state["hold_frames"] += 1
            if state["hold_frames"] > 12: 
                state["mission_idx"] = (m_idx + 1) % n
                state["step_idx"] = 0
                state["hold_frames"] = 0
                
                text_result.set_visible(False)
                text_result.get_bbox_patch().set_alpha(0)
                draw_map(ax_map) 
                line_trail.set_data([], [])
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