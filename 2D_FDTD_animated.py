import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Constants

eps0 = 8.8541878128e-12
mu0 = 1.256637062e-6
c = 1/np.sqrt(eps0*mu0)
imp0 = np.sqrt(mu0/eps0)
j_max = 500  #size of y
i_max = 500 #size of z
n_max = 2000  #size of t
j_source = 250 #space step of j_source
i_source = 250 #space step of i_source

# values for other than vakuum, epsilon (eps) also can be an array of permativities in the y direction for every step, mu also can be so

mu = mu0
eps = eps0*np.ones(shape=(i_max,j_max))
eps[100:200,100:200] = 10*eps0
material_prof = eps > eps0

v = 1/np.sqrt(eps*mu)
imp = np.sqrt(mu/eps)


E_x = np.zeros(shape=(i_max,j_max))
H_z = np.zeros(shape=(i_max,j_max))
H_y = np.zeros(shape=(i_max,j_max))

#can be replaced with just E_x and H_z and is not important but or simplicity, it can be better do define it

E_x_prev = np.zeros(shape=(i_max,j_max))
H_z_prev = np.zeros(shape=(i_max,j_max))
H_y_prev = np.zeros(shape=(i_max,j_max))

lambda_min = 350e-9  #minimum wavelength
dy = lambda_min/20
dz = lambda_min/20
S=0.5
dt = S/((np.sqrt(1/dy**2+1/dz**2))*c)

# Source function (for demonstration)
def Source_Func(t):
    tau = 30   # in time steps
    t_0 = tau*3     # delay for source to work
    lambda_0 = 550e-9  #defines the frequency of the source
    w0 = 2*np.pi*c/lambda_0 
    return np.exp(-(t-t_0)**2/tau**2)*np.sin(w0*t*dt)


#MATPLOT LIB ANIMATION PLOT SETUP

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(E_x.T, origin='lower', cmap='RdBu', interpolation='quadric',
               vmin=-0.05, vmax=0.05)
cb = plt.colorbar(im, ax=ax, label=r'$E_x$ amplitude')

# static overlay for material
overlay = ax.imshow(material_prof.T, origin='lower', cmap='Greys', alpha=0.15,
                    interpolation='nearest')
title = ax.set_title('t = 0')
ax.set_xlabel('y-index')
ax.set_ylabel('z-index')
plt.tight_layout()

# The time loop function
def fdtd_step(n):
    global E_x, H_y, H_z, E_x_prev, H_y_prev, H_z_prev
    
    E_x_old = E_x.copy()
    #MAXWELL EQUATIONS
    
    # Update magnetic field z
    H_z[:,:j_max-1] = H_z_prev[:,:j_max-1] + dt/(dy*mu0) * (E_x[:,1:j_max] - E_x[:,:j_max-1])
    H_z_prev = H_z
    
    # Update magnetic field y
    H_y[:i_max-1,:] = H_y_prev[:i_max-1,:] - dt/(dz*mu0) * (E_x[1:i_max,:] - E_x[:i_max-1,:])
    H_y_prev = H_y

    # Update electric field
    E_x[1:,1:] = E_x_prev[1:,1:] + dt/(dy*eps[1:,1:]) * (H_z[1:,1:] - H_z[1:,:j_max-1]) - dt/(dz*eps[1:,1:]) * (H_y[1:,1:] - H_y[:i_max-1,1:])
    E_x_prev = E_x

    #SOURCE
    # Add magnetic field source of H_z
    #H_z[i_source-1,j_source-1] -= Source_Func(n)/imp0
    #H_z_prev[i_source-1,j_source-1] = H_z[i_source-1,j_source-1]
    
    # Add magnetic field source H_y
    #H_y[i_source-1,j_source-1] -= Source_Func(n)/imp0
    #H_y_prev[i_source-1,j_source-1] = H_y[i_source-1,j_source-1]
    
    # Add electric field source E_x, If you want radial wave, only use E_x, for a directional wave, define H_y and H_z
    E_x[i_source,j_source] += Source_Func(n+1)
    E_x_prev[i_source,j_source] = E_x[i_source,j_source]


    #Mur 1st-order ABC (uses E_x_old as the previous step, because E_x_prev gets updated)
    S_y = c*dt/dy
    S_z = c*dt/dz
    coef_y = (S_y - 1)/(S_y + 1)
    coef_z = (S_z - 1)/(S_z + 1)

    # y-boundaries (j = 0, j = j_max-1)
    E_x[:, 0] = E_x_old[:, 1] + coef_y*(E_x[:, 1] - E_x_old[:, 0])
    E_x[:, j_max-1] = E_x_old[:, j_max-2] + coef_y*(E_x[:, j_max-2] - E_x_old[:, j_max-1])

    # z-boundaries (i = 0, i = i_max-1)
    E_x[0, :] = E_x_old[1, :] + coef_z*(E_x[1, :] - E_x_old[0, :])
    E_x[i_max-1, :] = E_x_old[i_max-2, :] + coef_z*(E_x[i_max-2, :] - E_x_old[i_max-1, :])

steps_per_frame = 1
def update(frame):
    # advance multiple FDTD steps per frame
    base_n = frame * steps_per_frame
    for s in range(steps_per_frame):
        fdtd_step(base_n + s)
    im.set_data(E_x.T)
    title.set_text(f't = {base_n + steps_per_frame}')
    return im, title, overlay

ani = FuncAnimation(fig, update, frames=n_max // steps_per_frame,
                    interval=20, blit=True)

plt.show()