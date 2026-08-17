import time
import matplotlib.pyplot as plt


time_steps = []
z_values = []
velocity_values = []
control_values = []
setpoint_values = []

# Desired state
z_desired = 1.0
v_desired = 0   # Setting desired velocity = 0 because we want the drone to get to the desired altitude and hover
a_desired = 0   # Setting desired acceleration = 0 because we want the drone to get to the desired altitude and hover

"""
Setting a desired velocity and acceleration becomes useful anytime the drone needs to follow deliberate motion rather than hover.

A delivery drone might have a trajectory commanding it to accelerate upward after takeoff, climb at a specified rate, 
decelerate as it reaches cruise altitude, and finally settle into level flight.

A camera drone could follow a carefully planned vertical trajectory 
so the camera rises smoothly at a constant speed rather than repeatedly accelerating toward new position setpoints.
"""

# Initial state
z_t = 0.0
v_t = 0.0

# PID gains
kp = 30
ki = 2.6
kd = 3

integral = 0.0

dt = 0.01

# Constants
g = 9.81
m_actual = 0.20
m_estimated = 0.15

max_force = 2.12


def drone_simplified_pid_controller(z_desired, z_t, v_t, kp, ki, kd, integral, dt):
    error = z_desired - z_t

    integral += error * dt

    # Since desired velocity = 0 (we want the drone to get to the desired altitude and hover):
    # derivative of altitude error ≈ -vertical velocity
    derivative = -v_t   # derivative according to derivation in notes is -(kd * v_t)

    control = m_estimated * (
        kp * error
        + ki * integral
        + kd * derivative
        + g
    )

    # Clamp to actuator limits (0 to 2.12N)
    control = min(max(0, control), max_force)

    return control, integral

def drone_full_pid_controller(z_desired, z_t, v_t, kp, ki, kd, integral, dt):
    error = z_desired - z_t

    integral += error * dt

    velocity_error = v_desired - v_t

    control = m_estimated * (
        kp * error
        + ki * integral
        + kd * velocity_error
        + g
        + a_desired
    )

    control = min(max(0, control), max_force)

    return control, integral


def process(force, z, velocity, dt):
    # Net vertical acceleration
    acceleration = force / m_actual - g

    # Integrate acceleration -> velocity
    velocity_new = velocity + acceleration * dt     # (m/s^2) * s = m/s

    # Integrate velocity -> position
    z_new = z + velocity_new * dt

    return z_new, velocity_new


if __name__ == "__main__":

    i = 0
    simulation_time = 10  # seconds
    while True:

        control, integral = drone_simplified_pid_controller(z_desired, z_t, v_t, kp, ki, kd, integral, dt)

        z_t, v_t = process(control, z_t, v_t, dt)

        time_steps.append(i * dt)
        z_values.append(z_t)
        velocity_values.append(v_t)
        control_values.append(control)
        setpoint_values.append(z_desired)

        time.sleep(dt)  # Simulate real-time by sleeping for dt seconds

        # in real time, the loop would run indefinitely, but for simulation purposes, we can stop after a certain time
        i += 1
        if i * dt > simulation_time:  # Run the simulation for 10 seconds
            break

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_steps, z_values, label="Altitude")
    plt.plot(
        time_steps,
        setpoint_values,
        linestyle="--",
        label="Desired Altitude"
    )
    plt.ylabel("Altitude (m)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, velocity_values)
    plt.ylabel("Velocity (m/s)")

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, control_values)
    plt.ylabel("Thrust (N)")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.show()