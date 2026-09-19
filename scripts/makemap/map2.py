from slam import *
import matplotlib.pyplot as plt

l = 20
dx = 0.05
n_side = int(l / dx)

grid = np.zeros((n_side, n_side), dtype=np.uint8)

sqr2meter=2

def draw_rect(grid,ul,sz):
    x0,y0=ul
    x0=int(sqr2meter*x0/dx)
    y0=int(sqr2meter*y0/dx)
    w,h=sz
    w=int(sqr2meter*w/dx)
    h=int(sqr2meter*h/dx)
    grid[x0:x0+w,y0:y0+h]=1

draw_rect(grid,(2,2),(5,2))
draw_rect(grid,(7.5,2),(1.5,2))
draw_rect(grid,(9,0),(1,1))

draw_rect(grid,(2,6),(2,2))
draw_rect(grid,(0,7.0),(3,1))
draw_rect(grid,(5.5,6),(3,2))

draw_rect(grid,(9,9),(1,1))
draw_rect(grid,(0,9),(1,1))

draw_rect(grid,(4.5,9.2),(0.75,0.8))

grid[:2,:]=1
grid[n_side-2:,:]=1
grid[:,:2]=1
grid[:,n_side-2:]=1


plt.imshow(grid)
plt.show()

gm=GridMap(grid,dx)
gm.write(DATA_DIR/Path("map2"))


