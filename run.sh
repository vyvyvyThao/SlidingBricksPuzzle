#!/bin/sh

if [ "$#" -lt 2 ]; then
    echo "An argument is missing"
    exit 1

else 
    python3 -c "
import sys
from SBP import processCommand
processCommand(sys.argv[1], sys.argv[2])
" "$1" "$2"
fi