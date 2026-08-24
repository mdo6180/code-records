import matplotlib.pyplot as plt



# Simulation history
time_steps = []
phi_values = []
p_values = []
phi_desired_values = []
p_desired_values = []
control_values = []
setpoint_values = []


# Desired altitude
phi_desired = 10.0

# Initial state
phi = 0.0
angular_velocity = 0.0

# Physical constants
Ixx_actual = 0.0035
Ixx_estimated = 0.0035

dt = 0.01


kp_phi = 3.0

def roll_attitude_controller(phi_desired, phi):
    error = phi_desired - phi
    p_desired = kp_phi * error
    return p_desired


kp_p = 5.0
ki_p = 0.0

angular_velocity_integral = 0.0
def roll_rate_controller(angular_velocity_desired, angular_velocity, angular_velocity_integral, dt):
    error = angular_velocity_desired - angular_velocity
    
    # Calculate what the integral WOULD become
    integral_candidate = angular_velocity_integral + error * dt

    # PI calculations...
    alpha_desired = (
        kp_p * error
        + ki_p * integral_candidate
    )

    tau_x = Ixx_estimated * alpha_desired

    angular_velocity_integral = integral_candidate

    return tau_x, angular_velocity_integral


def roll_process(tau_x, p, phi, dt):
    alpha_x = tau_x / Ixx_actual

    p += alpha_x * dt
    phi += p * dt

    return p, phi



for i in range(2000):
    angular_velocity_desired = roll_attitude_controller(phi_desired, phi)

    tau_x, angular_velocity_integral = roll_rate_controller(angular_velocity_desired, angular_velocity, angular_velocity_integral, dt)

    angular_velocity, phi = roll_process(tau_x, angular_velocity, phi, dt)

    time_steps.append(i * dt)
    phi_values.append(phi)
    p_values.append(angular_velocity)
    phi_desired_values.append(phi_desired)
    p_desired_values.append(angular_velocity_desired)
    control_values.append(tau_x)


plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time_steps, phi_values, label="Roll angle (phi)")
plt.plot(
    time_steps,
    phi_desired_values,
    linestyle="--",
    label="Desired roll angle"
)
plt.xlabel("Time (s)")
plt.ylabel("Roll angle (rad)")
plt.legend()
plt.title("Roll Angle (phi) vs Time (s)")

plt.subplot(3, 1, 2)
plt.plot(time_steps, p_values, label="Roll rate (p)")
plt.plot(
    time_steps,
    p_desired_values,
    linestyle="--",
    label="Desired roll rate"
)
plt.xlabel("Time (s)")
plt.ylabel("Roll rate (rad/s)")
plt.legend()
plt.title("Roll Rate (p) vs Time (s)")

plt.subplot(3, 1, 3)
plt.plot(time_steps, control_values, label="Control input (tau_x)")
plt.xlabel("Time (s)")
plt.ylabel("Control input (N*m)")
plt.legend()
plt.title("Control Input (tau_x) vs Time (s)")  

plt.tight_layout()
plt.show()