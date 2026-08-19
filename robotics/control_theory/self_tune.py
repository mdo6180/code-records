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

kp_z = 3.0

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

kp_v = 5.0
ki_v = 3.0

velocity_integral = 0.0


def velocity_controller(
    vz_desired,
    vz,
    integral,
    dt
):
    velocity_error = vz_desired - vz

    # Calculate what the integral WOULD become
    integral_candidate = integral + velocity_error * dt

    desired_acceleration = (
        kp_v * velocity_error
        + ki_v * integral_candidate
    )

    # Calculate UNSATURATED thrust first
    thrust_unsaturated = m_estimated * (
        g + desired_acceleration
    )

    # Clamp to physical actuator limits
    thrust = min(
        max(thrust_unsaturated, 0.0),
        max_force
    )

    # Anti-windup
    #
    # Accept the new integral unless:
    # 1. we're saturated high and error wants MORE thrust
    # 2. we're saturated low and error wants LESS thrust

    saturated_high = thrust_unsaturated > max_force
    saturated_low = thrust_unsaturated < 0.0

    pushing_higher = velocity_error > 0
    pushing_lower = velocity_error < 0

    if not (
        (saturated_high and pushing_higher)
        or
        (saturated_low and pushing_lower)
    ):
        integral = integral_candidate

    return thrust, integral


# -------------------------------------------------
# Drone physics
# -------------------------------------------------

def process(thrust, z, vz, dt):
    acceleration = thrust / m_actual - g

    vz = vz + acceleration * dt
    z = z + vz * dt

    # Simple ground-contact constraint
    if z < 0.0:
        z = 0.0
        vz = 0.0

    return z, vz


# -------------------------------------------------
# Simulation
# -------------------------------------------------

# Set vz=0.0, z=0.0, and then tune inner loop to achieve vz_desired for these values
#vz_desired = 0.1
#vz_desired = 0.3
#vz_desired = 0.5
#vz_desired = 0.7
#vz_desired = 0.9

# Set vz=0.0, z=0.0, and then tune outer loop to achieve z_desired for these values
z_desired = 0.2
#z_desired = 0.4
#z_desired = 0.6
#z_desired = 1.0
#z_desired = 5.0
#z_desired = 10.0

# Note:
# With z_desired = 0.2 and kp_z = 2.0, the drone initially descends
# below z = 0 because the controller calculates gravity compensation
# using m_estimated = 0.15 kg, while the actual mass is 0.20 kg.
#
# Consequently, the initial commanded thrust is less than the actual
# hover thrust, so the drone accelerates downward until the velocity
# controller's integral term compensates for the mass-model error.
#
# Increasing kp_z to 3.0 produces a larger initial desired velocity,
# which causes the velocity controller to request more thrust and
# greatly reduces the initial descent. However, this masks the mass
# estimation error rather than correcting its underlying cause.
# But this can still work as the velocity controller will correct itself much quicker, 
# thus allowing the drone to take off sooner.
# 
# However, the correct fix is to use the correct mass in the controller or to fix the process to disallow negative altitudes.

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