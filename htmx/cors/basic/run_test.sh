#!/bin/bash

# run the following command in terminal to execute this bash script:
# $ chmod +x test.sh
# $ ./test.sh

# Determine the operating system
OS_TYPE=$(uname)

# Define file paths of root and leaf services
FILEPATH_ROOT="root_service.py"
FILEPATH_LEAF="leaf_service.py"

# Define IP addresses and ports
IP_ROOT="127.0.0.1"
IP_LEAF="192.168.100.2"
PORT_ROOT="8000"
PORT_LEAF="8002"

# Check if the system is Linux or macOS and then define network interfaces
if [ "$OS_TYPE" = "Linux" ]; then
    echo "Running on Linux"
    INTERFACE="eth0"
elif [ "$OS_TYPE" = "Darwin" ]; then
    echo "Running on macOS"
    INTERFACE="en0"
else
    echo "Unsupported operating system: $OS_TYPE"
    exit 1
fi

# Configure the interface 
sudo ifconfig $INTERFACE alias $IP_LEAF netmask 255.255.255.0 up
ifconfig $INTERFACE

# Start pipelines
python $FILEPATH_ROOT $IP_ROOT $PORT_ROOT &
PID1=$!
python $FILEPATH_LEAF $IP_LEAF $PORT_LEAF &
PID2=$!

# Function to ping an IP address
ping_ip() {
    local ip=$1
    echo "Pinging $ip..."
    if ping -c 2 $ip > /dev/null 2>&1; then
        echo "$ip is reachable."
    else
        echo "$ip is not reachable."
    fi
}

# Ping the IP addresses
ping_ip $IP_ROOT
ping_ip $IP_LEAF

# Function to wait for background processes to complete (or until terminated by Ctrl+C)
cleanup() {
    echo "Stopping FastAPI servers..."
    wait $PID1
    wait $PID2
    echo "FastAPI servers stopped."
}

# Set up the trap to call the cleanup function on SIGINT (Ctrl+C)
trap cleanup SIGINT

# Wait for background processes to complete (or until terminated by Ctrl+C)
# Note: SIGINT is sent to root and leaf services when Ctrl+C is pressed
wait $PID1
wait $PID2

# remove an IP address from en0
echo "Removing IP address $IP_LEAF from $INTERFACE..."
sudo ifconfig $INTERFACE -alias $IP_LEAF

# Bring the interface down:
echo "Bringing $INTERFACE down..."
sudo ifconfig $INTERFACE down

echo "Done."