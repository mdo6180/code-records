import matplotlib.pyplot as plt


# Simulation history
time_steps = []
z_values = []
vz_values = []
vz_desired_values = []
control_values = []
setpoint_values = []


# Desired altitude
z_desired = 1.0

# Initial state
z = 0.0
vz = 0.0

# Physical constants
g = 9.81
m_actual = 0.20
m_estimated = 0.15
max_force = 2.12

dt = 0.01


# -------------------------------------------------
# Outer loop: altitude -> desired vertical velocity
# -------------------------------------------------

kp_z = 2.0

max_climb_rate = 1.0  # m/s


def altitude_controller(z_desired, z):
    altitude_error = z_desired - z

    vz_desired = kp_z * altitude_error

    # Don't allow the altitude controller to request
    # unreasonable vertical speeds.
    vz_desired = min(
        max(vz_desired, -max_climb_rate),
        max_climb_rate
    )

    return vz_desired


# -------------------------------------------------
# Inner loop: vertical velocity -> thrust
# -------------------------------------------------

kp_v = 4.0
ki_v = 2.0

velocity_integral = 0.0


def velocity_controller(
    vz_desired,
    vz,
    integral,
    dt
):
    velocity_error = vz_desired - vz

    integral += velocity_error * dt

    desired_acceleration = (
        kp_v * velocity_error
        + ki_v * integral
    )

    thrust = m_estimated * (
        g + desired_acceleration
    )

    thrust = min(max(thrust, 0.0), max_force)

    return thrust, integral


# -------------------------------------------------
# Drone physics
# -------------------------------------------------

def process(thrust, z, vz, dt):
    acceleration = thrust / m_actual - g

    # Integrate acceleration -> velocity
    vz = vz + acceleration * dt

    # Integrate velocity -> altitude
    z = z + vz * dt

    return z, vz


# -------------------------------------------------
# Simulation
# -------------------------------------------------

for i in range(2000):

    # OUTER LOOP
    vz_desired = altitude_controller(
        z_desired,
        z
    )

    # INNER LOOP
    control, velocity_integral = velocity_controller(
        vz_desired,
        vz,
        velocity_integral,
        dt
    )

    # Plant / drone dynamics
    z, vz = process(
        control,
        z,
        vz,
        dt
    )

    # Save data
    time_steps.append(i * dt)
    z_values.append(z)
    vz_values.append(vz)
    vz_desired_values.append(vz_desired)
    control_values.append(control)
    setpoint_values.append(z_desired)


# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time_steps, z_values, label="Altitude")
plt.plot(
    time_steps,
    setpoint_values,
    linestyle="--",
    label="Desired altitude"
)
plt.ylabel("Altitude (m)")
plt.legend()
plt.title("Altitude")

plt.subplot(3, 1, 2)
plt.plot(
    time_steps,
    vz_values,
    label="Actual vertical velocity"
)
plt.plot(
    time_steps,
    vz_desired_values,
    linestyle="--",
    label="Desired vertical velocity"
)
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.title("Vertical Velocity")

plt.subplot(3, 1, 3)
plt.plot(
    time_steps,
    control_values,
    label="Thrust"
)
plt.ylabel("Thrust (N)")
plt.xlabel("Time (s)")
plt.legend()
plt.title("Control Output")

plt.tight_layout()
plt.show()