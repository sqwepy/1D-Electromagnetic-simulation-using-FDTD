import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Constants

eps0 = 8.8541878128e-12
mu0 = 1.256637062e-6
c = 1/np.sqrt(eps0*mu0)
imp0 = np.sqrt(mu0/eps0)

j_max = 1000  #size of y
n_max = 2000  #size of t
j_source = 10 #space step of j_source

# values for other than vakuum, epsilon (eps) also can be an array of permativities in the y direction for every step, mu also can be so

mu = mu0
eps = np.ones(j_max)*eps0
eps[250:350] = 10*eps0
eps[650:750] = 20*eps0
v = 1/np.sqrt(eps*mu)
imp = np.sqrt(mu/eps)

material_prof = eps > eps0

E_x = np.zeros(j_max)
H_z = np.zeros(j_max)

#can be replaced with just E_x and H_z and is not important but or simplicity, it can be better do define it

E_x_prev = np.zeros(j_max)
H_z_prev = np.zeros(j_max)

lambda_min = 350e-9  #minimum wavelength
dx = lambda_min/20
dt = dx/c

# Source function (for demonstration)
def Source_Func(t):
    tau = 30   # in time steps
    t_0 = tau*3     # delay for source to work
    lambda_0 = 550e-9  #defines the frequency of the source
    w0 = 2*np.pi*c/lambda_0 
    return np.exp(-(t-t_0)**2/tau**2)*np.sin(w0*t*dt)

# Set up the figure and axis
fig, ax = plt.subplots()
line, = ax.plot(E_x, lw=2,color='lightblue')
ax.plot(material_prof, color='darkred')
ax.set_ylim([-2, 2])
ax.set_xlim([0, j_max])
ax.set_xlabel("Grid index j")
ax.set_ylabel("E-field amplitude")

# The update function for animation
def update(frame):
    global E_x, E_x_prev, H_z, H_z_prev

    # Update magnetic field
    H_z[j_max-1] = H_z_prev[j_max-2]
    H_z[:j_max-1] = H_z_prev[:j_max-1] + dt/(dx*mu0) * (E_x[1:j_max] - E_x[:j_max-1])
    H_z_prev[:] = H_z[:]

    # Add magnetic field source
    H_z[j_source-1] -= Source_Func(frame)/imp0
    H_z_prev[j_source-1] = H_z[j_source-1]

    # Update electric field
    E_x[0] = E_x_prev[1]
    E_x[1:] = E_x_prev[1:] + dt/(dx*eps[1:]) * (H_z[1:] - H_z[0:j_max-1])
    E_x_prev[:] = E_x[:]

    # Add electric field source
    E_x[j_source] += Source_Func(frame+1)
    E_x_prev[j_source] = E_x[j_source]

    # Update the line data
    line.set_ydata(E_x)

    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=n_max, interval=10, blit=True)

plt.show()
