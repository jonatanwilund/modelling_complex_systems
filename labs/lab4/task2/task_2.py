import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
from matplotlib.animation import FuncAnimation

#load the point cloud
df = pd.read_csv("points_lab04.txt", sep= " ", header = None)
vertices = df.to_numpy()

#bases: for vertices, row of associated vertex in vertices. For edges, ordered tuple (i,j), i<j of associated vertices.
#For faces, ordered triple (i,j,k), i<j<k of associated vertices


#build matrix of pointwise distances
def get_distances(points):
    differences = points[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(differences, axis = 2)
    
    return distances


#builds adjecency matrix for graph (entries are bools not 0,1)
def get_adjacency_matrix(distances, r):
    A = distances < r
    np.fill_diagonal(A, False)
    
    return A


#convert adjecency matrix to list of edges represented by tuples of row indeces for corresponding point in points matrix
def get_edges(adjacency_matrix):
    n = len(adjacency_matrix)
    edges = []
    
    #check only combinations of indeces for adjecency. Note combination itterates in ascending order.
    for i, j in combinations(range(n), 2):
        if adjacency_matrix[i,j]:
            edges.append((i,j))
        
    return edges

#convert adjecency matrix to list of faces represented by tripples of row indeces for corresponding point in points matrix
def get_faces(adjacency_matrix):
    n = len(adjacency_matrix)
    faces = []
    
    #check only combinations of indeces for adjecency. 
    for i, j, k in combinations(range(n), 3):
        if adjacency_matrix[i,j] and adjacency_matrix[i,k] and adjacency_matrix[j,k]:
            faces.append((i,j,k))
    
    return faces

def animate_complex(rs, points):
    distances = get_distances(points)
    fig, ax = plt.subplots()
    
    prev_edges = set()
    prev_faces = set()
    
    ax.scatter(points[:,0], points[:,1])
    ax.axis("equal")
    
    title = ax.set_title("r = 0 ")
    
    def update_frame(r):
        A = get_adjacency_matrix(distances,r)
        
        edges = set(get_edges(A))
        faces = set(get_faces(A))
        
        #plot edges
        for i,j in edges - prev_edges:
            x = [points[i,0], points[j,0]]
            y = [points[i,1], points[j,1]]
            ax.plot(x, y, color = "black")
            
        #plot faces
        for i,j,k in faces - prev_faces:
            x = [points[i,0], points[j,0], points[k,0]]
            y = [points[i,1], points[j,1], points[k,1]]
            ax.fill(x , y, color = "blue")
            
        prev_edges.update(edges)
        prev_faces.update(faces)
        beta_0, beta_1 = betti(points, edges, faces)
        title.set_text(f"r = {r:.4f}| $\\beta_0$ = {beta_0}, $\\beta_1$ = {beta_1}")
        
        
    anim = FuncAnimation(fig, update_frame, frames = rs, interval = 200)
    return anim


#displays a plot of the complex
def plot_complex(points, edges, faces):
    #plot vertices
    plt.scatter(points[:,0], points[:,1])
    
    #plot edges
    for i, j in edges:
        x = [points[i,0], points[j,0]]
        y = [points[i,1], points[j,1]]
        plt.plot(x, y, '-k')


    #plot faces
    for i, j, k in faces:
        x = [points[i,0], points[j,0], points[k,0]]
        y = [points[i,1], points[j,1], points[k,1]]
        plt.fill(x, y, color = "b")
            
    plt.axis('equal')
    

#gaussian elimination mod 2
def gaussian_m2(M):
    rows, cols = np.shape(M)
    
    rank = 0
    
    for col in range(cols):
        #find a pivot element
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
    
        #swap pivot and rank row 
        M[rank, :], M[pivot, :] = M[pivot, :].copy(), M[rank, :].copy()
        #add pivot to rows with 1's in column col
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row, :] = (M[row ,:] + M[rank, :]) % 2
                
        rank += 1
    
    return rank

def betti(points, edges, faces):
    n_vertices = len(points)
    n_edges = len(edges)
    n_faces = len(faces)
    
    #boundary operator matrices
    d_1 = np.zeros((n_vertices, n_edges))
    
    for column, (i, j) in enumerate(edges):
        d_1[i, column], d_1[j, column] = 1, 1
        
    #For d2 need to explicitly index edges via convention in the preamble
    edge_index = {e: i for i, e in enumerate(edges)}
    d_2 = np.zeros((n_edges, n_faces))
        
    for column, (i, j, k) in enumerate(faces):
        #note that eg (i,j) is an edge but not (j,i) by our convention
        for edge in [(i,j), (i,k), (j,k)]:
            index = edge_index[edge]
            d_2[index, column] = 1
        
    rank_d1 = gaussian_m2(d_1)
    rank_d2 = gaussian_m2(d_2)
    
    b_0 = n_vertices - rank_d1
    b_1 = n_edges - rank_d1 - rank_d2
    
    return(b_0,b_1)

def plot_betti(rs, points):
    b_0s = []
    b_1s = []
    
    distances = get_distances(points)
    
    for r in rs:
        A = get_adjacency_matrix(distances, r)
        edges = get_edges(A)
        faces = get_faces(A)
        
        b_0, b_1 = betti(points, edges, faces)
        b_0s.append(b_0)
        b_1s.append(b_1)
        
    fig, ax = plt.subplots()
    ax.step(rs, b_0s, where = "post", label=r"$\beta_0$")
    ax.step(rs, b_1s, where = "post", label=r"$\beta_1$")

    ax.set_xlabel("r")
    ax.set_ylabel("Betti numbers")
    ax.legend()
    
    return fig


def main(rs, points, plot = True, animate = True, save_animation = False, save_plot = False):
    distances = get_distances(points)
    if animate:
        anim = animate_complex(rs, points)
        plt.show()
        
        if save_animation:
            anim.save("Vietoris_Rips.gif", writer="pillow", fps=5)
    if plot:
        fig = plot_betti(rs, points)
        plt.show()
        
        if save_plot:
            fig.savefig("betti_plot.png", dpi=300, bbox_inches="tight")
        
if __name__ == "__main__":    
    distances = get_distances(vertices)
    #remove the first element of the rs (trivial 0)
    rs = np.unique(distances)[1:]
    main(rs, vertices, plot = True, animate = True, save_plot = True)
    
