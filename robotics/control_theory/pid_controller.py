# https://cookierobotics.com/051/

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# in this situation, we are simulating a quadrotor with an actual mass of 0.2 kg, 
# but the controller is designed for a quadrotor with an estimated mass of 0.15 kg.
# the drone will hover at a lower altitude than desired because the controller is not providing enough thrust to counteract the actual weight of the drone.
# this is where a PID controller would be useful, 
# as it would be able to adjust the control input based on the error between the desired and actual altitude, 
# allowing the drone to reach and maintain the desired altitude despite the mass mismatch.

# Constants
g = 9.81   # Gravitational acceleration (m/s^2)
m_actual = 0.2
m_estimated = 0.15

"""
# dx/dt = f(t, x)
# 
# t     : Current time (seconds), scalar
# x     : Current state, [z, vz]
# return: First derivative of state, [vz, az]
def xdot(t, x):
    # Desired z, vz, az
    z_des  = 1

    # desired vertical velocity (m/s) is set to 0 for hover (no vertical movement), 
    # if you want to move at a certain velocity e.g., like in situation with manual controller input, 
    # you can set this to a positive or negative value respectively
    vz_des = 0
    az_des = 0

    # PD Controller (input, u)

    # Proportional gain (kp) is set to 30, which is a common value for quadrotor control.
    # if kp is too low (kp=1), the system will be slow to respond and will take a long time to reach the desired state.
    # if kp is too high (kp=100), the system will be very responsive but will overshoot and oscillate around the desired state making it harder to hover.
    kp = 30

    # derivative gain (kv) is set to 3, which is a common value for quadrotor control.
    # this is an underdamped system, so the kv value is set to a value that will provide a good balance between responsiveness and stability.
    # kv=1 will increase the overshoot and oscillations, while kv=5 will make the system more stable but slower to respond 
    # and thus will take a longer time to reach the desired state (see what happens when kv=20).
    kv = 3

    u  = m_estimated * (az_des + kp * (z_des - x[0]) + kv * (vz_des - x[1]) + g)
    
    # Clamp to actuator limits (0 to 2.12N)
    u = min(max(0, u), 2.12)
    
    # Quadrotor dynamics (dx/dt = xdot = [vz, az])
    return [x[1], u/m_actual - g]


simulation_time = 10 # seconds
x0     = [0, 0] # Initial state, [z0, vz0]
t_span = [0, simulation_time] # Simulation time (seconds), [from, to]


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
    u  = m_estimated * (az_des + kp * (z_des - x[0]) + kv * (vz_des - x[1]) + g)
    
    # Clamp to actuator limits (0 to 2.12N)
    u = min(max(0, u), 2.12)
    return u

u = [force(t, x) for t, x in zip(sol.t, sol.y.T)]

# Create a figure with 3 rows and 1 column
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

# Plot altitude: z vs t
ax1.plot(sol.t, sol.y[0], '-o', color="blue")
ax1.axhline(y=1, color='gray', linestyle='--', label='Desired Altitude (1 m)')
ax1.set_title("Quadrotor Altitude vs Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("z(t): Altitude (m)")

# Plot velocity: vz vs t
ax2.plot(sol.t, sol.y[1], '-o', color="red")
ax2.set_title("Quadrotor Vertical Velocity vs Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("vz(t): Vertical Velocity (m/s)")

# Plot control input: u vs t
ax3.plot(sol.t, u, '-o', color="green")
ax3.set_title("Quadrotor Control Input Force vs Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("u(t): Control Input Force (N)")

plt.show()
"""


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
    ki = 2.6

    # State
    z  = x[0]
    vz = x[1]
    iz = x[2]  # accumulated altitude error

    # Errors
    ez  = z_des - z
    evz = vz_des - vz

    u  = m_estimated * (az_des + kp * ez + kv * evz + ki * iz + g)
    
    # Clamp to actuator limits (0 to 2.12N)
    u = min(max(0, u), 2.12)

    # Dynamics
    az = u / m_actual - g
    
    # Quadrotor dynamics (dx/dt = xdot = [vz, az])
    return [vz, az, ez]  # include the integral of altitude error


# Initial state:
# z = 0 m
# vz = 0 m/s
# integral error = 0
simulation_time = 10 # seconds
x0     = [0, 0, 0] # Initial state, [z0, vz0, iz0]
t_span = [0, simulation_time] # Simulation time (seconds), [from, to]


# Solve for the states, x(t) = [z(t), vz(t), iz(t)]
sol = solve_ivp(xdot, t_span, x0)

def force(t, x):
    z  = x[0]
    vz = x[1]
    iz = x[2]

    z_des  = 1
    vz_des = 0
    az_des = 0

    ez  = z_des - z
    evz = vz_des - vz

    kp = 30
    kv = 3
    ki = 2.6

    u = m_estimated * (
        az_des
        + kp * ez
        + kv * evz
        + ki * iz
        + g
    )

    return min(max(0, u), 2.12)


u = [
    force(t, x)
    for t, x in zip(sol.t, sol.y.T)
]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    4, 1,
    figsize=(7, 11),
    sharex=True
)

ax1.plot(sol.t, sol.y[0])
ax1.axhline(1, linestyle="--")
ax1.set_ylabel("Altitude (m)")
ax1.set_title("Altitude")

ax2.plot(sol.t, sol.y[1])
ax2.set_ylabel("Velocity (m/s)")
ax2.set_title("Vertical Velocity")

ax3.plot(sol.t, sol.y[2])
ax3.set_ylabel("Integral error")
ax3.set_title("Accumulated Altitude Error")

ax4.plot(sol.t, u)
ax4.set_ylabel("Thrust (N)")
ax4.set_xlabel("Time (s)")
ax4.set_title("Control Input")

plt.tight_layout()
plt.show()
