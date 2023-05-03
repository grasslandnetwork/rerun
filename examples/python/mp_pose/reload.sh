#!/bin/bash

# Define the command to run the Python file
COMMAND_TO_RUN="python3 main.py --video_path \$1 --connect"

# Define a function to run when the script is interrupted
function cleanup {
    echo "Cleaning up..."
    exit
}

# Set the trap to catch the interrupt signal and run the cleanup function
trap cleanup INT

# Run the command initially
$COMMAND_TO_RUN

# Monitor all .py files in the current directory for changes and re-run the command when a change is detected
while true; do
    inotifywait -q -e modify --exclude '.*\.swp' --exclude '.*\.swx' $(find . -name '*.py') || true
    $COMMAND_TO_RUN
done
