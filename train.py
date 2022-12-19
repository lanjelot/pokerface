
from genericpath import isfile
import os
from PIL import Image
import cv2
import subprocess
import time

BOARD1_VALUE_REGION = (810, 405, 830, 465)
BOARD2_VALUE_REGION = (932, 405, 952, 465)
BOARD3_VALUE_REGION = (1048, 405, 1080, 468)
BOARD4_VALUE_REGION = (1172, 405, 1205, 468)
BOARD5_VALUE_REGION = (1296, 405, 1329, 468)

CARD_VALUES = "23456789TJQKA"


MYCARD1_VALUE_REGION = (1110, 765, 1155, 855)
MYCARD2_VALUE_REGION = (1240, 750, 1280, 840)

def train_mycards(d='/home/seb/screencaps-pixel'):
    for f in sorted(os.listdir(d)):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)

        # image = image.crop(MYCARD1_VALUE_REGION)
        image = image.crop(MYCARD2_VALUE_REGION)

        subprocess.Popen(['geeqie', image_path])
        while True:
            v = input('Card value? ')
            v = v.upper()
            if v in CARD_VALUES:
                break
        subprocess.Popen(['killall', 'geeqie'])

        if not v:
            continue
        t = '/home/seb/cv/mycard2/%s.png' % v
        if os.path.isfile(t):
            continue
        image.save(t)

def train_board(d='/home/seb/screencaps-all/'):
    for f in sorted(os.listdir(d)):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)

        image1 = image.crop(BOARD1_VALUE_REGION)
        image2 = image.crop(BOARD2_VALUE_REGION)
        # image3 = image.crop(BOARD3_VALUE_REGION)

        images = [image1, image2, ]
        subprocess.Popen(['geeqie', image_path])
        for image in images:
            while True:
                v = input('Card value? ')
                v = v.upper()
                if v in CARD_VALUES:
                    break
            subprocess.Popen(['killall', 'geeqie'])

            if not v:
                continue
            t = '/home/seb/cv/%s.png' % v
            if os.path.isfile(t):
                continue
            image.save(t)

if __name__ == '__main__':
    train_mycards()