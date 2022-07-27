#!/bin/bash

while :; do
  sleep 5
  adb exec-out screencap -p > ~/screencaps/$(date +%s).png
done
