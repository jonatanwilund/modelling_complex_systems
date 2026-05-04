import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button 

N = 20
update_interval = 200
t = 0
states = {"resting": 0, "ready": 1, "firing": 2}

#Usage: Left click cells to set them to resting (white). Right click to set them to firing (red).
#Left click a resting cell to set it to ready (green). Space bar to pause/unpause simulation, esc to reset board

def get_firing_neighbors(cells, i,j):

    firing_neighbors = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbor_i = (i + di) % N
            neighbor_j = (j + dj) % N
            
            neighbor = cells[neighbor_i,neighbor_j]
            
            if neighbor == 2:
                firing_neighbors += 1
                
    return firing_neighbors

def step(cells):
    next_gen_cells = cells.copy()
    for i in range(N):
        for j in range(N):
            cell = cells[i,j]
            if cell == 0:
                next_gen_cells[i,j] = 1
            
            elif cell == 1:
                neighbors = get_firing_neighbors(cells, i,j)
                
                if neighbors == 2:
                    next_gen_cells[i,j] = 2
                else:
                    next_gen_cells[i,j] = 1
                    
            elif cell == 2:
                next_gen_cells[i,j] = 0
    return next_gen_cells
    
class Automata:
    def __init__(self):
        #initialize background as all ready
        self.cells = np.ones((N,N), dtype = int)
        self.is_running = False
        self.is_mouse_down = False
        #colors in RGB
        self.color_dict = {
            0 : (1, 1, 1), #resting - white
            1: (90/255, 200/255, 40/255), #ready - green
            2: (210/255, 40/255, 40/255) #firing - red
        }
        self.draw_state = 0
        
        self.fig = plt.figure(figsize = (6,6))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])
        self.image = self.ax.imshow(
            self._cell_colors(),
            interpolation = "nearest",
            origin = "upper"
            )
        
        #clear major ticks and set minors to draw grid
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticks(np.arange(-0.5, N, 1), minor = True)
        self.ax.set_yticks(np.arange(-0.5, N, 1), minor = True)
        self.ax.grid(which = "minor", color = (0.5,0.5,0.5), linewidth = 0.5)
        self.ax.tick_params(which = "minor", length = 0)
        
        #key presses
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_drag)
        
        self.timer = self.fig.canvas.new_timer(interval = update_interval)
        self.timer.add_callback(self._update)
        self.timer.start()
        
    def _cell_colors(self):
        color_grid = np.zeros((N,N,3),dtype = "float")
        for state, color in self.color_dict.items():
            mask = (self.cells == state)
            color_grid[mask] = color
            
        return color_grid
        
    def _on_key(self, event):
        if event.key == " ":
            self.is_running = not self.is_running
        elif event.key == "right":
            if not self.is_running:
                #progress the system one step
                self.cells = step(self.cells)
                self._refresh()
        elif event.key == "escape":
            self.is_running = False
            self.cells = np.ones((N, N), dtype=int)
            self._refresh()
            
            
    def _on_click(self, event):
        if event.inaxes == self.ax:
            self.is_mouse_down = True
            coordinates = self._get_cell_coordinates(event)
            if coordinates == None:
                return
            if self.cells[coordinates] != 0:
                self.draw_state = 0
            else:
                self.draw_state = 1
            self._draw_cell(event)
            #color the cell at mouse position
            
    def _on_release(self, event):
        self.is_mouse_down = False
    
    def _on_drag(self, event):
        if self.is_mouse_down and event.inaxes == self.ax:
            #color the cell at the mouse position
            self._draw_cell(event)
            
            
    def _get_cell_coordinates(self, event):
        if event.inaxes != self.ax or event.xdata == None or event.ydata == None:
            return None
        i, j = round(event.ydata), round(event.xdata)
        if 0 <= i < N and 0 <= j < N:
            return i, j
        return None
    
    def _draw_cell(self, event):
        coordinates = self._get_cell_coordinates(event)
        if not self.is_mouse_down or coordinates is None:
            return
        if event.button == 1:
            self.cells[coordinates] = self.draw_state
        if event.button == 3:
            self.cells[coordinates] = 2
        self._refresh()
        
    def _refresh(self):
        self.image.set_data(self._cell_colors())
        self.fig.canvas.draw_idle()
        
    def _update(self):
        if self.is_running:
            self.cells = step(self.cells)
            self._refresh()


def main():
    automata = Automata()
    plt.show()
    
if __name__ == "__main__":
    main()
                
                
            
