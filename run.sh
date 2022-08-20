#!/bin/bash

stty -echo
python3 pokerface.py "$@"
stty echo
echo
