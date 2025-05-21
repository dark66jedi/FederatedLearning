#!/bin/bash
set -e

SWAPFILE=/swapfile
SWAPSIZE=4G

echo "Creating a ${SWAPSIZE} swap file at ${SWAPFILE}..."

# Create swap file
sudo fallocate -l $SWAPSIZE $SWAPFILE || \
sudo dd if=/dev/zero of=$SWAPFILE bs=1M count=4096 status=progress

# Set permissions
sudo chmod 600 $SWAPFILE

# Setup swap area
sudo mkswap $SWAPFILE

# Enable swap
sudo swapon $SWAPFILE

# Backup fstab before editing
sudo cp /etc/fstab /etc/fstab.bak

# Add swap entry to /etc/fstab if not present
if ! grep -q "^$SWAPFILE" /etc/fstab; then
    echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab
fi

echo "Swap of size $SWAPSIZE created and enabled successfully."
echo "Current swap status:"
swapon --show
