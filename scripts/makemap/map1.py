from slam import *
import matplotlib.pyplot as plt

l = 20
dx = 0.05
n_side = int(l / dx)

grid = np.zeros((n_side, n_side), dtype=np.uint8)


# Define some walsl for the room
x = np.arange(l)
y = np.arange(l)
x, y = np.meshgrid(x, y)
big_cross = (x == l // 2) | (y == l / 2)
doors = ((2 < x) & (x < 7)) | ((13 < x) & (x < 18)) | ((13 < y) & (y < 18))|(y<7)
bottomblock = (15 < x) & (15 < y)
boolmap = (big_cross | bottomblock) & (~doors)

# Transform to other shape
X = (np.arange(n_side)*dx).astype(np.int32)
Y = (np.arange(n_side)*dx).astype(np.int32)
X, Y = np.meshgrid(X, Y)
grid[boolmap[X, Y]] = True

# Add some cool balls
def dist(r:tuple,r0:tuple):
    x,y=r
    x0,y0=r0
    return np.sqrt((x-x0)**2+(y-y0)**2)

X = (np.arange(n_side,dtype=np.float32)*dx)
Y = (np.arange(n_side,dtype=np.float32)*dx)
X, Y = np.meshgrid(X, Y)
grid[ dist((X,Y),(5,4))<0.9 ]=1
grid[ dist((X,Y),(12,4))<0.7 ]=1

grid[ dist((X,Y),(0,20))<0.9 ]=1
grid[ dist((X,Y),(3,15))<0.9 ]=1
grid[ dist((X,Y),(13,2))<0.9 ]=1
grid[ dist((X,Y),(12,13))<0.9 ]=1

grid[0,:]=1
grid[:,0]=1
grid[n_side-1,:]=1
grid[:,n_side-1]=1

plt.imshow(grid)
plt.show()

gm=GridMap(grid,dx)
gm.write(DATA_DIR/Path("map1"))


