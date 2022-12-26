#!/usr/bin/env python

import os
from PIL import Image
from time import sleep
from sys import argv

def convert(filepath):
    with open(filepath, 'rb') as f:
        image = Image.frombytes('RGBA', (2340, 1080), f.read())
    image.save(filepath.replace('raw', 'png'))
    os.unlink(filepath)

if len(argv) == 1:
    d = './screencaps-auto'
else:
    d = argv[1]
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
