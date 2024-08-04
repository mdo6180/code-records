# Define IP addresses and ports
IP_SERVER="127.0.0.1"
IP_CLIENT="192.168.100.2"
PORT_SERVER="8001"
PORT_CLIENT="8002"

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

# remove an IP address from en0
echo "Removing IP address $IP_CLIENT from $INTERFACE..."
sudo ifconfig $INTERFACE -alias $IP_CLIENT

# Bring the interface down:
echo "Bringing $INTERFACE down..."
sudo ifconfig $INTERFACE down

echo "Done."