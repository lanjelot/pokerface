#!/usr/bin/env python

import os
from PIL import Image
from time import sleep

def convert(filepath):
    with open(filepath, 'rb') as f:
        image = Image.frombytes('RGBA', (2340, 1080), f.read())
    image.save(filepath.replace('raw', 'png'))
    os.unlink(filepath)

d = '/home/seb/screencaps-auto'
while True:
    for filename in sorted(os.listdir(d)):
        if not filename.endswith('.raw'):
            continue
        print(filename)
        filepath = os.path.join(d, filename)
        try:
            convert(filepath)
        except:
            break

    sleep(1)
