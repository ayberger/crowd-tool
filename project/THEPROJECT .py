import numpy as np
import pygame
import random
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from queue import Queue, deque
import networkx as nx
import os
import time
import math
import json
import statistics


STEP_LENGTH_M = 0.76


frames_global = 0

agents_lock = threading.Lock()
buckets_global = {}


def clamp(val, lo, hi):
    return max(lo, min(val, hi))

def inflate_obstacles(grid, margin=0):  
    rows, cols = grid.shape
    inflated = grid.copy()
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] in (1, 3):
                for dr in range(-margin, margin+1):
                    for dc in range(-margin, margin+1):
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            inflated[rr, cc] = 1
    return inflated


def map_editor():
    r"""
    Opens an interactive map editor window.
    Left-click on a cell to cycle its value: 
        0 (=empty) →1 (wall) →2 (exit) →4 (spawn) →0.
    Right-click twice to draw a straight line of the currently
    selected block type between two points.
    Press 'S' to enter/exit spawn-region mode: define a rectangular
    spawn region by clicking two corners, then specify the number
    of agents for that region.
    Click “Done” to finish. Borders are enforced as walls
    (unless already exit/spawn), and the grid is saved to:
    C:/Users/Ayberk/OneDrive/Masaüstü/project/maps/<name>.txt
    plus spawn regions to <name>_regions.json
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    maps_dir   = os.path.join(script_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    
    root = tk.Tk(); root.withdraw()
    action = simpledialog.askstring("Map Editor",
        "Action? (new / load / delete)", initialvalue="new")
    root.destroy()

    existing = sorted(f for f in os.listdir(maps_dir) if f.endswith(".txt"))
    grid     = None
    map_name = None
    spawn_regions = []  

    if action == "load" and existing:
        
        root = tk.Tk(); root.withdraw()
        idx = simpledialog.askinteger(
            "Load Map",
            "Which map?\n" +
            "\n".join(f"{i+1}: {n}" for i,n in enumerate(existing)),
            minvalue=1, maxvalue=len(existing)
        )
        root.destroy()
        map_name = existing[idx-1]
        grid    = np.loadtxt(os.path.join(maps_dir, map_name), dtype=int)
        
        base = os.path.splitext(map_name)[0]
        regions_file = os.path.join(maps_dir, f"{base}_regions.json")
        if os.path.exists(regions_file):
            with open(regions_file,'r') as f:
                spawn_regions = json.load(f)

    elif action == "delete" and existing:
        
        root = tk.Tk(); root.withdraw()
        idx = simpledialog.askinteger(
            "Delete Map",
            "Which map delete?\n" +
            "\n".join(f"{i+1}: {n}" for i,n in enumerate(existing)),
            minvalue=1, maxvalue=len(existing)
        )
        root.destroy()
        to_del = existing[idx-1]
        os.remove(os.path.join(maps_dir, to_del))
        
        base = os.path.splitext(to_del)[0]
        rf = os.path.join(maps_dir, f"{base}_regions.json")
        if os.path.exists(rf): os.remove(rf)
        print(f"Deleted {to_del} and its region settings")
        return

    else:
        
        root = tk.Tk(); root.withdraw()
        rows = simpledialog.askinteger("Map Editor", "Rows?", initialvalue=20, minvalue=5, maxvalue=100)
        cols = simpledialog.askinteger("Map Editor", "Cols?", initialvalue=40, minvalue=5, maxvalue=100)
        root.destroy()
        grid = np.zeros((rows, cols), dtype=int)

    rows, cols = grid.shape
    
    base_size = 30
    subdivisions = 3
    cell_size = base_size // subdivisions   

    win_w = cols * cell_size
    win_h = rows * cell_size + 50

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Map Editor – L-click cycle [0→1→2→4→0], R-click twice to draw line, 'S' for spawn region")
    font = pygame.font.SysFont(None, 36)
    done_btn = pygame.Rect(10, rows*cell_size+5, win_w-20, 40)

    line_start = None
    line_type  = None
    region_mode = False
    region_start = None

    def draw_line(a, b, val):
        
        r0, c0 = a; r1, c1 = b
        dr = abs(r1-r0); dc = abs(c1-c0)
        sr = 1 if r1>=r0 else -1
        sc = 1 if c1>=c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            grid[r,c] = val
            if (r,c)==(r1,c1): break
            e2 = err*2
            if e2 > -dc:
                err -= dc; r += sr
            if e2 <  dr:
                err += dr; c += sc

    editing = True
    while editing:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
                
                region_mode = not region_mode
                region_start = None
                print(f"Spawn-region mode {'on' if region_mode else 'off'}")
            if ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                if my < rows*cell_size:
                    r, c = my//cell_size, mx//cell_size
                    if region_mode:
                        
                        if ev.button == 1:
                            if region_start is None:
                                region_start = (r, c)
                                print("Spawn region corner 1 set")
                            else:
                                region_end = (r, c)
                                
                                root = tk.Tk(); root.withdraw()
                                count = simpledialog.askinteger(
                                    "Spawn Count",
                                    "Number of agents for this region?",
                                    initialvalue=1, minvalue=1
                                )
                                root.destroy()
                                r0, c0 = region_start
                                r1, c1 = region_end
                                spawn_regions.append([r0, c0, r1, c1, count])
                                
                                for rr in range(min(r0,r1), max(r0,r1)+1):
                                    for cc in range(min(c0,c1), max(c0,c1)+1):
                                        grid[rr, cc] = 4
                                region_start = None
                                print(f"Added spawn region {(r0,c0)} to {(r1,c1)} count={count}")
                    else:
                        if ev.button == 1:
                            
                            grid[r,c] = {0:1, 1:2, 2:4}.get(grid[r,c], 0)
                        elif ev.button == 3:
                            
                            if line_start is None:
                                line_start = (r,c)
                                line_type  = grid[r,c]
                            else:
                                draw_line(line_start, (r,c), line_type)
                                line_start = None
                elif done_btn.collidepoint(mx,my) and ev.button==1:
                    editing = False

        
        screen.fill((220,220,220))
        for rr in range(rows):
            for cc in range(cols):
                val = grid[rr,cc]
                col = {0:(255,255,255),1:(0,0,0),2:(0,255,0),4:(0,0,255)}[val]
                rect = pygame.Rect(cc*cell_size, rr*cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, col, rect)
                pygame.draw.rect(screen, (200,200,200), rect, 1)
        
        if line_start:
            lr,lc = line_start
            hl = pygame.Rect(lc*cell_size, lr*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255,0,0), hl, 3)
        
        for (r0, c0, r1, c1, _) in spawn_regions:
            x = min(c0,c1)*cell_size; y = min(r0,r1)*cell_size
            w = (abs(c1-c0)+1)*cell_size; h = (abs(r1-r0)+1)*cell_size
            pygame.draw.rect(screen, (0,0,200), pygame.Rect(x,y,w,h), 2)
        
        pygame.draw.rect(screen, (100,100,100), done_btn)
        text = font.render("Done", True, (255,255,255))
        screen.blit(text, text.get_rect(center=done_btn.center))

        pygame.display.flip()
        pygame.time.delay(30)

    
    for c in range(cols):
        if grid[0,c] not in (2,4):       grid[0,c] = 1
        if grid[rows-1,c] not in (2,4):  grid[rows-1,c] = 1
    for r in range(rows):
        if grid[r,0] not in (2,4):       grid[r,0] = 1
        if grid[r,cols-1] not in (2,4):  grid[r,cols-1] = 1

    
    root = tk.Tk(); root.withdraw()
    name = simpledialog.askstring(
        "Save Map",
        f"Save as name (without .txt) [{map_name or ''}]:",
        initialvalue=(map_name or "")
    )
    root.destroy()
    if name:
        fname = name if name.endswith(".txt") else f"{name}.txt"
        out = os.path.join(maps_dir, fname)
        np.savetxt(out, grid, fmt="%d")
        
        base = os.path.splitext(fname)[0]
        regions_file = os.path.join(maps_dir, f"{base}_regions.json")
        with open(regions_file, 'w') as f:
            json.dump(spawn_regions, f, indent=2)
        
        main_txt = os.path.join(script_dir, "map.txt")
        np.savetxt(main_txt, grid, fmt="%d")
        live_regions = os.path.join(script_dir, "spawn_regions.json")
        with open(live_regions, 'w') as f:
            json.dump(spawn_regions, f)
        print(f"Map saved to maps/{fname} with {len(spawn_regions)} spawn region(s)")

    pygame.quit()



def fire_editor(grid, cell_size):
    
    rows, cols = grid.shape
    button_h  = 50
    win_w     = cols * cell_size
    win_h     = rows * cell_size + button_h

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Fire Editor – click to toggle fire, then Done")
    font = pygame.font.SysFont(None, 36)
    done_rect = pygame.Rect(10, rows*cell_size + 5, win_w - 20, button_h - 10)

    
    fire_mask = [[False]*cols for _ in range(rows)]
    editing = True
    while editing:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                editing = False
                break
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if my < rows * cell_size:
                    r, c = my // cell_size, mx // cell_size
                    
                    if grid[r, c] == 0:
                        fire_mask[r][c] = not fire_mask[r][c]
                elif done_rect.collidepoint(mx, my):
                    editing = False
                    break

       
        screen.fill((255,255,255))
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                if   val == 1: color = (  0,   0,   0)   
                elif val == 2: color = (  0, 255,   0)   
                else:           color = (200, 200, 200)  
                pygame.draw.rect(screen, color,
                                 (c*cell_size, r*cell_size, cell_size, cell_size))
                
                if fire_mask[r][c]:
                    pygame.draw.rect(screen, (255,  0,   0),
                                     (c*cell_size, r*cell_size, cell_size, cell_size))
                
                pygame.draw.rect(screen, (100,100,100),
                                 (c*cell_size, r*cell_size, cell_size, cell_size), 1)

        
        pygame.draw.rect(screen, (100,100,100), done_rect)
        txt = font.render("Done", True, (255,255,255))
        screen.blit(txt, txt.get_rect(center=done_rect.center))

        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()
   
    return [(r,c) for r in range(rows) for c in range(cols) if fire_mask[r][c]]




def load_graph_map():
    
    map_path = r"C:\Users\Ayberk\OneDrive\Masaüstü\project\map.txt"
    print("Loading map from:", map_path)
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Error: {map_path} not found.")
    with open(map_path, "r") as f:
        data = np.loadtxt(f, dtype=int)
    rows, cols = data.shape
    G = nx.Graph()
    node_positions = {}
    for r in range(rows):
        for c in range(cols):
            if data[r, c] in (0, 2, 3, 4): 
                node_positions[(r, c)] = True
    for (r, c) in node_positions.keys():
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in node_positions:
                G.add_edge((r, c), (r + dr, c + dc))
    return G, data


def bfs_path(inflated_grid, start, goal):
    rows, cols = inflated_grid.shape
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    if inflated_grid[start[0], start[1]] in (1, 3) or inflated_grid[goal[0], goal[1]] in (1, 3):
        return []
    visited = {start}
    parent = {}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1]

        cr, cc = current
        for dr, dc in moves:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if inflated_grid[nr, nc] in (1, 3) or (nr, nc) in visited: continue
            if dr != 0 and dc != 0:
                if inflated_grid[cr+dr, cc] == 1 or inflated_grid[cr, cc+dc] == 1:
                    continue
            visited.add((nr, nc))
            parent[(nr, nc)] = current
            queue.append((nr, nc))
    return []


def find_nearest_exit(grid, start):
    rows, cols = grid.shape
    if grid[start[0], start[1]] == 2:
        return start
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    visited = set([start])
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if grid[r, c] == 2:
            return (r, c)
        for dr, dc in moves:
            rr, cc = r+dr, c+dc
            if 0 <= rr < rows and 0 <= cc < cols:
                if (rr, cc) not in visited and grid[rr, cc] not in (1, 3):
                    visited.add((rr, cc))
                    queue.append((rr, cc))
    return None


def agent_thread(agent, agents, agents_lock, move_queue, cell_size, move_speed,
                 original_grid, inflated_grid, finished_agents):
    
    with agents_lock:
        agent["start_time"] = time.time()
        agent["replan_count"] = 0

    
    exit_goal = agent.get("exit_cell")
    allow_shortcut = agent.get("allow_shortcut", False)

    speed_pps = cell_size * 1000.0 / move_speed
    time.sleep(agent.get("spawn_delay", 0))
    time.sleep(2)
    last_time = time.time()
    rows, cols = original_grid.shape
    k_corner = 1.0
    k_wall   = 2.0
    k_agent  = 2.5

    while True:
       
        prev_display = agent["display_pos"]
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        with agents_lock:
            if not agent["grid_path"]:
                
                agent["exit_time"] = agent["ideal_exit_time"]
                finished_agents.append(agent)
                if agent in agents:
                    agents.remove(agent)
                move_queue.put(agent)
                break
            target = agent["grid_path"][0]
            target_center = (target[1]*cell_size + cell_size/2, target[0]*cell_size + cell_size/2)
            current_pos = agent["display_pos"]
            others = [o for o in agents if o is not agent]

        
        is_exit_step = (target == agent["exit_cell"])
        if not is_exit_step:
            with agents_lock:
                occ = sum(1 for o in agents
                    if o is not agent and o["pos"] == target)
       
            free_dirs = sum(
                1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= target[0]+dr < rows
                and 0 <= target[1]+dc < cols
                and original_grid[target[0]+dr, target[1]+dc] != 1
            )        
            threshold = 1 if free_dirs <= 2 else 2

            if occ >= threshold:
                current_cell = agent["pos"]
                exit_cell    = agent["exit_cell"]

                alt = []
    
                if (0 <= current_cell[0] < rows and
                    0 <= current_cell[1] < cols and
                    0 <= exit_cell[0]    < rows and
                    0 <= exit_cell[1]    < cols):

                    temp_grid = inflated_grid.copy()
                    temp_grid[target[0], target[1]] = 1
                    alt = bfs_path(temp_grid, current_cell, exit_cell)

                if alt:
                    with agents_lock:
                        agent["grid_path"]     = alt[1:]
                        agent["replan_count"] += 1
                    continue

    
                with agents_lock:
                    agent["display_pos"] = prev_display
                stuck_time = 0.0
                time.sleep(0.033)
                continue

        
        dx = target_center[0] - current_pos[0]
        dy = target_center[1] - current_pos[1]

        
        d2 = dx*dx + dy*dy

       
        arrival = max(cell_size * 0.4, 5)
        arrival2 = arrival * arrival

        
        if d2 < arrival2:
            with agents_lock:
                agent["grid_path"].pop(0)
            continue

        
        if d2 > 0:
            inv_dist = 1.0 / math.sqrt(d2)   
            dir_x = dx * inv_dist
            dir_y = dy * inv_dist
            dist  = 1.0 / inv_dist            
        else:
       
            dir_x = dir_y = 0.0
            dist  = 0.0


        
        
        nearby = []
        my_r, my_c = agent["pos"]
        bg = buckets_global     
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nearby.extend(bg.get((my_r+dr, my_c+dc), []))


        
        rep_a    = [0.0, 0.0]
        inf_r    = cell_size * 1.0
        inf_r2   = inf_r * inf_r
        cx_pos, cy_pos = current_pos

        
        if len(nearby) > 1:
            for o in nearby:
                if o is agent:
                    continue
                ox, oy = o["display_pos"]
                vx, vy = cx_pos - ox, cy_pos - oy
                d2 = vx*vx + vy*vy
                if 0 < d2 < inf_r2:
                    d  = math.sqrt(d2)
                    f  = (inf_r - d) / d
                    invd = 1.0 / d
                    rep_a[0] += f * (vx * invd)
                    rep_a[1] += f * (vy * invd)

        
        rep_c = [0.0, 0.0]
        ci = cell_size * 0.8
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r0 = my_r + dr
                c0 = my_c + dc
                if 0 <= r0 < rows and 0 <= c0 < cols and original_grid[r0, c0] in (1, 3):
                    corners = [
                        (c0 * cell_size,     r0 * cell_size),
                        (c0 * cell_size + cell_size, r0 * cell_size),
                        (c0 * cell_size,     r0 * cell_size + cell_size),
                        (c0 * cell_size + cell_size, r0 * cell_size + cell_size)
                    ]
                    for cx, cy in corners:
                        vx, vy = cx_pos - cx, cy_pos - cy
                        d0 = math.hypot(vx, vy)
                        if 0 < d0 < ci:
                            f = (ci - d0) / d0
                            rep_c[0] += f * (vx / d0)
                            rep_c[1] += f * (vy / d0)

        
        rep_w = [0.0, 0.0]
        sd = max(cell_size / 4, 6)
        for rr in range(max(0, my_r - 2), min(rows, my_r + 3)):
            for cc in range(max(0, my_c - 2), min(cols, my_c + 3)):
                if original_grid[rr, cc] in (1, 3):
                    left, right = cc*cell_size, (cc+1)*cell_size
                    top, bottom= rr*cell_size, (rr+1)*cell_size
                    nx0 = clamp(cx_pos, left, right)
                    ny0 = clamp(cy_pos, top, bottom)
                    d0 = math.hypot(cx_pos - nx0, cy_pos - ny0)
                    if 0 < d0 < sd:
                        f = (sd - d0) / sd
                        rep_w[0] += f * ((cx_pos - nx0) / d0)
                        rep_w[1] += f * ((cy_pos - ny0) / d0)

               

        
        density = len(nearby)
        density_thresh = 8  
        if density > density_thresh:
            k_agent_eff = k_agent * (density / density_thresh)
        else:
            k_agent_eff = k_agent

        
        cx_comb = dir_x + k_agent_eff * rep_a[0] + k_corner * rep_c[0] + k_wall * rep_w[0]
        cy_comb = dir_y + k_agent_eff * rep_a[1] + k_corner * rep_c[1] + k_wall * rep_w[1]

        
        angle = random.uniform(-0.2, 0.2)  
        ca, sa = math.cos(angle), math.sin(angle)
        fx = cx_comb * ca - cy_comb * sa
        fy = cx_comb * sa + cy_comb * ca

        
        norm = math.hypot(fx, fy)
        if norm:
            final_dir = (fx / norm, fy / norm)
        else:
            final_dir = (0.0, 0.0)

        
        step = speed_pps * dt
        if step > dist:
            step = dist
        new_x = cx_pos + final_dir[0] * step
        new_y = cy_pos + final_dir[1] * step

        
        r = int(new_y // cell_size)
        c = int(new_x // cell_size)
        r = max(0, min(rows - 1, r))
        c = max(0, min(cols - 1, c))

        if original_grid[r, c] in (1, 3):
            new_x, new_y = prev_display
            r, c = agent["pos"]

        with agents_lock:
            agent["display_pos"] = (new_x, new_y)
            agent["pos"]         = (r, c)
            move_queue.put(agent)

        time.sleep(0.02)



def visualize_simulation(grid, agents, move_queue, cell_size):
    global buckets_global
    global frames_global
    pygame.init()
    screen_w = cell_size * grid.shape[1]
    screen_h = cell_size * grid.shape[0]
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("School Building Crowd Simulation")
    clock = pygame.time.Clock()
    frame_count = 0
    running = True
    frame_count_local = 0
    t0 = time.time()
    while running:
        with agents_lock:
            bg = {}
            for o in agents:
                cell = o["pos"]     
                bg.setdefault(cell, []).append(o)
            buckets_global = bg
        screen.fill((255,255,255))
        frame_count += 1
        frame_count_local += 1 
        fire_color = (255,69,0) if frame_count % 30 < 15 else (255,0,0)
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                if val == 1:
                    color = (0,0,0)
                elif val == 2:
                    color = (0,255,0)
                elif val == 3:
                    color = fire_color
                elif val == 4:
                    color = (255,255,255)
                else:
                    color = (255,255,255)
                pygame.draw.rect(screen, color, (c*cell_size, r*cell_size, cell_size, cell_size))
        
        density_threshold = 3
        density = {}
        with threading.Lock():
            agents_copy = agents[:]
        for ag in agents_copy:
            col = int(ag["display_pos"][0] // cell_size)
            row = int(ag["display_pos"][1] // cell_size)
            density[(row, col)] = density.get((row, col), 0) + 1
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        for (row, col), count in density.items():
            if count >= density_threshold:
                cx = col*cell_size + cell_size/2
                cy = row*cell_size + cell_size/2
                radius = 40 + (count - density_threshold)*6
                pygame.draw.circle(overlay, (255, 0, 255, 150), (int(cx), int(cy)), int(radius))
        screen.blit(overlay, (0, 0))
        
        while not move_queue.empty():
            move_queue.get()
        
        with threading.Lock():
            for agent in agents:
                px, py = agent["display_pos"]
                pygame.draw.circle(screen, (0,0,255), (int(px), int(py)), max(cell_size//3,6))
        
        pygame.display.flip()
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    elapsed = time.time() - t0
    if elapsed > 0.0:
        frames_global = frame_count_local / elapsed
    else:
        frames_global = 0.0

        
    pygame.quit()


def generate_report(finished_agents, total_agents, simulation_duration):
   
    global frames_global

    
    num_finished = len(finished_agents)
    num_unfinished = total_agents - num_finished

    exit_times = [
        ag["exit_time"]   
        for ag in finished_agents
        if "exit_time" in ag
    ]
    if exit_times:
        min_exit   = min(exit_times)
        max_exit   = max(exit_times)
        avg_exit   = statistics.mean(exit_times)
        med_exit   = statistics.median(exit_times)
        stdev_exit = statistics.stdev(exit_times) if len(exit_times)>1 else 0.0
    else:
        min_exit = max_exit = avg_exit = med_exit = stdev_exit = 0.0

    
    replan_counts = [ag.get("replan_count", 0) for ag in finished_agents]
    if replan_counts:
        min_replan = min(replan_counts)
        max_replan = max(replan_counts)
        avg_replan = statistics.mean(replan_counts)
        stdev_replan = statistics.stdev(replan_counts) if len(replan_counts) > 1 else 0.0
    else:
        min_replan = max_replan = avg_replan = stdev_replan = 0.0

    
    throughput = num_finished / simulation_duration if simulation_duration > 0 else 0.0

    
    categories = {}
    for ag in finished_agents:
        cat = ag.get("category", "unknown")
        categories.setdefault(cat, []).append( ag.get("ideal_exit_time", 0.0))
    cat_stats = {}
    for cat, times in categories.items():
        count = len(times)
        cat_stats[cat] = {
            "count": count,
            "avg_exit_time": statistics.mean(times) if times else 0.0,
            "max_exit_time": max(times) if times else 0.0
        }

    
    avg_fps = frames_global

    
    lines = []
    lines.append("Simulation Report")
    lines.append("-----------------")
    lines.append(f"Total agents created:   {total_agents}")
    lines.append(f"Agents that exited:      {num_finished}")
    lines.append(f"Agents still in simulation: {num_unfinished}")
    lines.append(f"Simulation duration (s): {simulation_duration:.3f}")
    lines.append(f"Throughput (agents/s):   {throughput:.3f}")
    lines.append(f"Average FPS during rendering loop:  {avg_fps:.2f}")
    lines.append("")
    lines.append("Exit‐Time Statistics (s):")
    et = {
        "min": min_exit, "max": max_exit,
        "mean": avg_exit, "median": med_exit,
        "stdev": stdev_exit
    }
    lines.append(f"  • Min     = {et['min']:.3f}")
    lines.append(f"  • Max     = {et['max']:.3f}")
    lines.append(f"  • Mean    = {et['mean']:.3f}")
    lines.append(f"  • Median  = {et['median']:.3f}")
    lines.append(f"  • Stdev   = {et['stdev']:.3f}")
    lines.append("")
    lines.append("Replanning Counts (per agent):")
    rp = {
        "min": min_replan, "max": max_replan,
        "mean": avg_replan, "stdev": stdev_replan
    }
    lines.append(f"  • Min     = {rp['min']}")
    lines.append(f"  • Max     = {rp['max']}")
    lines.append(f"  • Mean    = {rp['mean']:.3f}")
    lines.append(f"  • Stdev   = {rp['stdev']:.3f}")
    lines.append("")
    lines.append("Breakdown by Category:")
    for cat, stats in cat_stats.items():
        lines.append(f"  • {cat}:")
        lines.append(f"      – Count       = {stats['count']}")
        lines.append(f"      – Avg Exit (s)= {stats['avg_exit_time']:.3f}")
        lines.append(f"      – Max Exit (s)= {stats['max_exit_time']:.3f}")
    lines.append("")

    
    report_data = {
        "total_agents": total_agents,
        "num_finished": num_finished,
        "num_unfinished": num_unfinished,
        "simulation_duration_s": round(simulation_duration, 3),
        "throughput_agents_per_s": round(throughput, 3),
        "average_fps": round(avg_fps, 1),
        "exit_time_stats_s": et,
        "replan_stats": rp,
        "category_breakdown": cat_stats
    }
    return "\n".join(lines), report_data



def main():
    
    root = tk.Tk()
    root.withdraw()
    design_new = messagebox.askyesno("Map Editor", "Would you like to design a new map?")
    root.destroy()
    if design_new:
        map_editor()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    
    regions_file = os.path.join(script_dir, "spawn_regions.json")
    spawn_regions = []
    if os.path.exists(regions_file):
        with open(regions_file, "r") as rf:
            spawn_regions = json.load(rf)

    
    _, temp_grid = load_graph_map()
    rows, cols = temp_grid.shape

    root = tk.Tk()
    root.withdraw()
    seed = simpledialog.askinteger(
            "Simulation Seed",
            "Enter integer seed for replayability (or leave blank):",
            initialvalue=int(time.time())
        )

    root.destroy()

    while True:

        random.seed(seed)
    
        graph, original_grid = load_graph_map()

        rows, cols = original_grid.shape

        pygame.init()
        screen_info = pygame.display.Info()
        sw = screen_info.current_w - 100
        sh = screen_info.current_h - 100
        cell_size = min(sw // cols, sh // rows)
        cell_size = max(cell_size, 10)
        pygame.quit() 


       
        manual = messagebox.askyesno(
            "Fire Placement",
            "Place fires manually? (Yes = choose on map; No = random)"
        )
        if manual:
            fire_cells = fire_editor(original_grid, cell_size)
        else:
            empty_cells = [
                (r,c)
                for r in range(rows)
                for c in range(cols)
                if original_grid[r,c] == 0
            ]
            num_fires  = 3
            fire_cells = random.sample(empty_cells,
                                  min(num_fires, len(empty_cells)))

        
        for r, c in fire_cells:
            original_grid[r, c] = 3


        print(f"🔥 Fires at {fire_cells}")



        inflated_grid = inflate_obstacles(original_grid, margin=0)
    
        
        spawn_cells = [(r,c) for r in range(rows) for c in range(cols) if original_grid[r,c] == 4]

        real_cell_length_m = STEP_LENGTH_M
    
        pygame.init()
        screen_info = pygame.display.Info()
        sw = screen_info.current_w - 100
        sh = screen_info.current_h - 100
        cell_size = min(sw // cols, sh // rows)
        cell_size = max(cell_size, 10)
    
        agents = []
        finished_agents = []  
        move_queue = Queue()

        
        agents = []
        for start_pos in spawn_cells:
            exit_cell = find_nearest_exit(original_grid, start_pos)
            if not exit_cell:
                continue
            path = bfs_path(inflated_grid, start_pos, exit_cell)
            if not path:
                continue

        
            category = random.choice(["child", "adult", "elderly"])
            if category == "child":
                
                ms_per_cell = (real_cell_length_m * 1000.0) / 1.19
            elif category == "elderly":
                
                ms_per_cell = (real_cell_length_m * 1000.0) / 1.04
            else:
                
                ms_per_cell = (real_cell_length_m * 1000.0) / 1.34

            ag_speed = int(ms_per_cell)

            ideal_steps      = len(path) - 1
            ideal_exit_time  = (ideal_steps * ag_speed) / 1000.0


            sx = start_pos[1]*cell_size + cell_size/2
            sy = start_pos[0]*cell_size + cell_size/2
            agents.append({
                "pos":         start_pos,
                "display_pos": (sx, sy),
                "grid_path":   path[1:],
                "exit_cell":   exit_cell,
                "spawn_delay": 0.0,
                "category":    category,
                "move_speed":  ag_speed,
                "start_time":  None,
                "exit_time":   None,
                "replan_count":0,
                "ideal_exit_time": ideal_exit_time
            })

        total_agents = len(agents)
        print(f"Total agents created: {total_agents}")

    
        simulation_start = time.time()
        threads = []
        for ag in agents:
            
            t = threading.Thread(
                target=agent_thread,
                args=(ag, agents, agents_lock, move_queue,
                    cell_size, ag["move_speed"],
                    original_grid, inflated_grid, finished_agents)
            )
            threads.append(t)
            t.start()

    
        visualize_simulation(original_grid, agents, move_queue, cell_size)
    
        for t in threads:
            t.join()
        simulation_end = time.time()
        simulation_duration = simulation_end - simulation_start

        report_str, report_data = generate_report(finished_agents, total_agents, simulation_duration)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"simulation_report_{timestamp}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_str)
        print("Report generated:", report_filename)
        print(report_str)
        
        json_filename = f"simulation_report_{timestamp}.json"
        with open(json_filename, "w", encoding="utf-8") as f_json:
            json.dump(report_data, f_json, indent=2)

        again = messagebox.askyesno(
            "Repeat Simulation",
            "Run the same scenario again?"
        )
        if not again:
            break

if __name__ == "__main__":
    main()
