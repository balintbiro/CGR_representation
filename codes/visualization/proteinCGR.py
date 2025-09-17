import numpy as np
import matplotlib.pyplot as plt

x0, y0=0, 0
r=1
angles=np.linspace(start=0,stop=2*np.pi,num=20)

x=x0+r*np.sin(angles)
y=y0+r*np.cos(angles)

plt.figure(figsize=(6,6))
plt.plot(x, y,"o")
plt.gca().set_aspect('equal')
