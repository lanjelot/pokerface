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

TEMPLATES_MYCARD1_VALUE = {}
for v in CARD_VALUES:
    tv = cv2.imread('./cv/mycard1/%s.png' % v, 0)
    TEMPLATES_MYCARD1_VALUE[v] = tv

TEMPLATES_MYCARD1_SUITS = {}
for v in CARD_SUITS:
    tv = cv2.imread('./cv/mycard1/%s.png' % v, 0)
    TEMPLATES_MYCARD1_SUITS[v] = tv

TEMPLATES_MYCARD2_VALUE = {}
for v in CARD_VALUES:
    tv = cv2.imread('./cv/mycard2/%s.png' % v, 0)
    TEMPLATES_MYCARD2_VALUE[v] = tv

TEMPLATES_MYCARD2_SUITS = {}
for v in CARD_SUITS:
    tv = cv2.imread('./cv/mycard2/%s.png' % v, 0)
    TEMPLATES_MYCARD2_SUITS[v] = tv

TEMPLATES_BOARD_VALUE = {}
for v in CARD_VALUES:
    tv = cv2.imread('./cv/board/%s.png' % v, 0)
    TEMPLATES_BOARD_VALUE[v] = tv

TEMPLATES_BOARD_SUIT = {}
for v in CARD_SUITS:
    tv = cv2.imread('./cv/board/%s.png' % v, 0)
    TEMPLATES_BOARD_SUIT[v] = tv

REGION_MYCARD1_VALUE = (1100, 760, 1165, 860)
REGION_MYCARD2_VALUE = (1230, 745, 1290, 845)
REGION_MYCARD1_SUIT = (1115, 865, 1175, 945)
REGION_MYCARD2_SUIT = (1230, 855, 1290, 935)

REGION_BOARD1_VALUE = (805, 400, 850, 470)
REGION_BOARD2_VALUE = (925, 400, 970, 470)
REGION_BOARD3_VALUE = (1050, 400, 1095, 470)
REGION_BOARD4_VALUE = (1175, 400, 1210, 470)
REGION_BOARD5_VALUE = (1295, 400, 1330, 470)

REGION_BOARD1_SUIT = (805, 470, 850, 520)
REGION_BOARD2_SUIT = (925, 470, 970, 520)
REGION_BOARD3_SUIT = (1050, 470, 1095, 520)
REGION_BOARD4_SUIT = (1175, 470, 1210, 520)
REGION_BOARD5_SUIT = (1295, 470, 1330, 520)


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
    v1 = match_symbol(image, REGION_MYCARD1_VALUE, TEMPLATES_MYCARD1_VALUE)
    v2 = match_symbol(image, REGION_MYCARD2_VALUE, TEMPLATES_MYCARD2_VALUE)

    s1 = match_symbol(image, REGION_MYCARD1_SUIT, TEMPLATES_MYCARD1_SUITS)
    s2 = match_symbol(image, REGION_MYCARD2_SUIT, TEMPLATES_MYCARD2_SUITS)

    return v1+s1+v2+s2

def read_board(image):
    v1 = match_symbol(image, REGION_BOARD1_VALUE, TEMPLATES_BOARD_VALUE)
    v2 = match_symbol(image, REGION_BOARD2_VALUE, TEMPLATES_BOARD_VALUE)
    v3 = match_symbol(image, REGION_BOARD3_VALUE, TEMPLATES_BOARD_VALUE)
    v4 = match_symbol(image, REGION_BOARD4_VALUE, TEMPLATES_BOARD_VALUE)
    v5 = match_symbol(image, REGION_BOARD5_VALUE, TEMPLATES_BOARD_VALUE)

    s1 = match_symbol(image, REGION_BOARD1_SUIT, TEMPLATES_BOARD_SUIT)
    s2 = match_symbol(image, REGION_BOARD2_SUIT, TEMPLATES_BOARD_SUIT)
    s3 = match_symbol(image, REGION_BOARD3_SUIT, TEMPLATES_BOARD_SUIT)
    s4 = match_symbol(image, REGION_BOARD4_SUIT, TEMPLATES_BOARD_SUIT)
    s5 = match_symbol(image, REGION_BOARD5_SUIT, TEMPLATES_BOARD_SUIT)
    
    board = v1+s1, v2+s2, v3+s3, v4+s4, v5+s5

    b = ''
    for c in board:
        if c not in DECK:
            break
        b += c
    return b

def chunk(s, bs):
  return [s[i:i + bs] for i in range(0, len(s), bs)]

def pp_cards(cards, pfx=''):
    print('%s%s' % (pfx, ' '.join(chunk(cards, 2))))

XY_FOLD = (1000, 1000)
XY_CALL = (1500, 1000)
XY_ANY = (2000, 1000)

def tap(xy):
    # time.sleep(random.uniform(0.8, 1.2)) #2.2))
    x, y = [str(n) for n in xy]
    subprocess.check_output(['adb', 'shell', 'input', 'tap', x, y])

SBAR_Y = 930
SBAR_X = [
    1990,
    2035,
    2080,
    2120
]

def can_check(image):
    return image.getpixel((1447, 1010)) == (255, 255, 255, 255)

def how_strong(image):
    strength = 0
    for x in SBAR_X:
        pixel = image.getpixel((x, SBAR_Y))
        # print(x, '->', pixel)
        if pixel[0] > 20:
            strength += 1
            continue
    return strength

def act(image):
    if image.getpixel((1915, 1010)) == (255, 255, 255, 255): # BET or RAISE available
        return

    h = how_strong(image)
    print('%d/4' % h)
    if h > 0:
        tap(XY_ANY)
    elif can_check(image):
        tap(XY_CALL)
    else:
        tap(XY_FOLD)

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

            act(image)
            continue

        cards = read_mycards(image)
        if '??' in cards:
             continue
        
        if '?' in cards:
            image.save('/home/seb/screencaps-nocards/%d.png' % time.time())
            continue 

        if prev_cards != cards:
            pp_cards(cards, '=> ')
            prev_cards = cards

            act(image)
            continue

if __name__ == '__main__':
    poll_table()

