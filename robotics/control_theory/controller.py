# https://cookierobotics.com/051/

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Constants
g = 9.81   # Gravitational acceleration (m/s^2)
m = 0.18   # Mass (kg)


# dx/dt = f(t, x)
# 
# t     : Current time (seconds), scalar
# x     : Current state, [z, vz]
# return: First derivative of state, [vz, az]
def xdot(t, x):
    # Desired z, vz, az
    z_des  = 1
    vz_des = 0
    az_des = 0

    # PD Controller (input, u)
    kp = 30
    kv = 3
    u  = m * (az_des + kp * (z_des - x[0]) + kv * (vz_des - x[1]) + g)
    
    # Clamp to actuator limits (0 to 2.12N)
    u = min(max(0, u), 2.12)
    
    # Quadrotor dynamics (dx/dt = xdot = [vz, az])
    return [x[1], u/m - g]


x0     = [0, 0] # Initial state, [z0, vz0]
t_span = [0, 5] # Simulation time (seconds), [from, to]


# Solve for the states, x(t) = [z(t), vz(t)]
sol = solve_ivp(xdot, t_span, x0)

# Solve for the control input, u(t)
def force(t, x):
    # Desired z, vz, az
    z_des  = 1
    vz_des = 0
    az_des = 0

    # PD Controller (input, u)
    kp = 30
    kv = 3
    u  = m * (az_des + kp * (z_des - x[0]) + kv * (vz_des - x[1]) + g)
    
    # Clamp to actuator limits (0 to 2.12N)
    u = min(max(0, u), 2.12)
    return u

u = [force(t, x) for t, x in zip(sol.t, sol.y.T)]

# Create a figure with 3 rows and 1 column
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

# Plot altitude: z vs t
ax1.plot(sol.t, sol.y[0], 'k-o', color="blue")
ax1.set_title("Quadrotor Altitude vs Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("z(t): Altitude (m)")

# Plot velocity: vz vs t
ax2.plot(sol.t, sol.y[1], 'k-o', color="red")
ax2.set_title("Quadrotor Vertical Velocity vs Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("vz(t): Vertical Velocity (m/s)")

# Plot control input: u vs t
ax3.plot(sol.t, u, 'k-o', color="green")
ax3.set_title("Quadrotor Control Input Force vs Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("u(t): Control Input Force (N)")

plt.show()
