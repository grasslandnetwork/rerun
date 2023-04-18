#!/bin/bash

# Activate conda environment
source activate rerun_io_4_gln


# Define the command to run the Python file
COMMAND_TO_RUN="python3 main.py --video_path $1 --connect"

# Run the command initially
$COMMAND_TO_RUN

# Monitor all .py files in the current directory for changes and re-run the command when a change is detected
while true; do
    inotifywait -q -e modify --exclude '.*\.swp' --exclude '.*\.swx' $(find . -name '*.py')
    $COMMAND_TO_RUN
done

