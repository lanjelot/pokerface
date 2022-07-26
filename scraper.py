import time
import os
import subprocess
import io
import random
import cv2
import numpy
from PIL import Image

CARD_VALUES = "23456789TJQKA"
CARD_SUITS = "shdc"
DECK = [x + y for x in CARD_VALUES for y in CARD_SUITS]

TEMPLATES_MYCARD1 = {}
for v in CARD_VALUES:
    tv = cv2.imread('/home/seb/cv/mycard1/%s.png' % v, 0)
    TEMPLATES_MYCARD1[v] = tv

TEMPLATES_MYCARD1_SUITS = {}
for v in CARD_SUITS:
    tv = cv2.imread('/home/seb/cv/mycard1/%s.png' % v, 0)
    TEMPLATES_MYCARD1_SUITS[v] = tv

TEMPLATES_MYCARD2 = {}
for v in CARD_VALUES:
    tv = cv2.imread('/home/seb/cv/mycard2/%s.png' % v, 0)
    TEMPLATES_MYCARD2[v] = tv

TEMPLATES_MYCARD2_SUITS = {}
for v in CARD_SUITS:
    tv = cv2.imread('/home/seb/cv/mycard2/%s.png' % v, 0)
    TEMPLATES_MYCARD2_SUITS[v] = tv

TEMPLATES_VALUES = {}
for v in CARD_VALUES:
    tv = cv2.imread('/home/seb/cv/%s.png' % v, 0)
    TEMPLATES_VALUES[v] = tv

TEMPLATES_SUITS = {}
for v in CARD_SUITS:
    tv = cv2.imread('/home/seb/cv/%s.png' % v, 0)
    TEMPLATES_SUITS[v] = tv

MYCARD1_VALUE_REGION = (1100, 760, 1165, 860)
MYCARD2_VALUE_REGION = (1230, 745, 1290, 845)
MYCARD1_SUIT_REGION = (1115, 865, 1175, 945)
MYCARD2_SUIT_REGION = (1230, 855, 1290, 935)

BOARD1_VALUE_REGION = (805, 400, 850, 470)
BOARD2_VALUE_REGION = (925, 400, 970, 470)
BOARD3_VALUE_REGION = (1050, 400, 1095, 470)
BOARD4_VALUE_REGION = (1175, 400, 1210, 470)
BOARD5_VALUE_REGION = (1295, 400, 1330, 470)

BOARD1_SUIT_REGION = (805, 470, 850, 520)
BOARD2_SUIT_REGION = (925, 470, 970, 520)
BOARD3_SUIT_REGION = (1050, 470, 1095, 520)
BOARD4_SUIT_REGION = (1175, 470, 1210, 520)
BOARD5_SUIT_REGION = (1295, 470, 1330, 520)


def test_screencaps(d='/home/seb/screencaps-board-bak/'):
    for i, f in enumerate(sorted(os.listdir(d))):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)

        actual = read_board(image)
        # actual = read_mycards(image)
        pp_cards(actual)

        subprocess.Popen(['geeqie', image_path])
        input()
        subprocess.Popen(['killall', 'geeqie'])

def match_symbol(image, region, templates):
    image = image.crop(region)
    # image.save('/tmp/needle.png')
    image_gray = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2GRAY)
    best_score = 0
    best_match = '?'
    for value, template in templates.items():
        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = numpy.where(res >= 0.8)
        # print(value, loc)
        score = len(loc[0])+len(loc[1])
        if score > best_score:
            best_score = score
            best_match = value
    return best_match

def read_mycards(image):    
    v1 = match_symbol(image, MYCARD1_VALUE_REGION, TEMPLATES_MYCARD1)
    v2 = match_symbol(image, MYCARD2_VALUE_REGION, TEMPLATES_MYCARD2)

    s1 = match_symbol(image, MYCARD1_SUIT_REGION, TEMPLATES_MYCARD1_SUITS)
    s2 = match_symbol(image, MYCARD2_SUIT_REGION, TEMPLATES_MYCARD2_SUITS)

    return v1+s1+v2+s2

def read_board(image):
    v1 = match_symbol(image, BOARD1_VALUE_REGION, TEMPLATES_VALUES)
    v2 = match_symbol(image, BOARD2_VALUE_REGION, TEMPLATES_VALUES)
    v3 = match_symbol(image, BOARD3_VALUE_REGION, TEMPLATES_VALUES)
    v4 = match_symbol(image, BOARD4_VALUE_REGION, TEMPLATES_VALUES)
    v5 = match_symbol(image, BOARD5_VALUE_REGION, TEMPLATES_VALUES)

    s1 = match_symbol(image, BOARD1_SUIT_REGION, TEMPLATES_SUITS)
    s2 = match_symbol(image, BOARD2_SUIT_REGION, TEMPLATES_SUITS)
    s3 = match_symbol(image, BOARD3_SUIT_REGION, TEMPLATES_SUITS)
    s4 = match_symbol(image, BOARD4_SUIT_REGION, TEMPLATES_SUITS)
    s5 = match_symbol(image, BOARD5_SUIT_REGION, TEMPLATES_SUITS)
    
    board = v1+s1, v2+s2, v3+s3, v4+s4, v5+s5

    b = ''
    for c in board:
        if c in DECK:
            b += c
        else:
            b += '..'
    return b #''.join(c for c in board if c in DECK)


def chunk(s, bs):
  return [s[i:i + bs] for i in range(0, len(s), bs)]

def pp_cards(cards, pfx=''):
    print('%s%s' % (pfx, ' '.join(chunk(cards, 2))))

XY_FOLD = (1000, 1000)
XY_CALL = (1500, 1000)
XY_ANY = (2000, 1000)

def tap(xy):
    time.sleep(random.uniform(0.8, 2.2))
    x, y = [str(n) for n in xy]
    subprocess.check_output(['adb', 'shell', 'input', 'tap', x, y])

def poll_table():
    prev_cards = ''
    prev_board = ''

    while True:
        out = subprocess.check_output(['adb', 'exec-out', 'screencap', '-p'])
        image = Image.open(io.BytesIO(out))
        board = read_board(image)
        
        if prev_board != board:
            image.save('/home/seb/screencaps-auto/%d.png' % time.time())
            pp_cards(board, '')
            prev_board = board
            continue

        cards = read_mycards(image)
        if '??' in cards:
             continue
        
        if '?' in cards:
            image.save('/home/seb/screencaps-nocards/%d.png' % time.time())
        #     cards = register_unknown(cards, image)        

        if prev_cards != cards:
            pp_cards(cards, '=> ')
            prev_cards = cards
            # tap(XY_FOLD)
            continue

        # tap(XY_CALL)

if __name__ == '__main__':
    poll_table()

#TODO
# always click on check/fold if hand strength shit
