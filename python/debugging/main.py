def calculate_force(mass, acceleration):
    force = mass * acceleration
    # breakpoint()  # This is where the debugger will pause execution
    return force

mass = 10
acceleration = 5

force = calculate_force(mass, acceleration)

print(force)