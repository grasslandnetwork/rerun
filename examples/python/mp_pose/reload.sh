#!/bin/bash

# Activate conda environment
source activate rerun_io_4_gln

# Set the path to the Python file you want to monitor
FILE_TO_MONITOR="main.py"

# Define the command to run the Python file
COMMAND_TO_RUN="python3 $FILE_TO_MONITOR --video_path /home/ubuntu/projects/raspberrypi3b_4_22.jpg --connect"

# Run the command initially
$COMMAND_TO_RUN

# Monitor the file for changes and re-run the command when a change is detected
while true; do
    inotifywait -e modify $FILE_TO_MONITOR
    $COMMAND_TO_RUN
done

