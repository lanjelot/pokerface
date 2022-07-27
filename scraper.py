from genericpath import isfile
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

    c1, c2 = v1+s1, v2+s2
    if c1 in DECK and c2 in DECK:
        return c1+c2

    filepath = '/home/seb/screencaps-failcards/%d.png' % time.time()
    image.save(filepath)

    print('Failed to read mycards %s %s -> %s' % (c1, c2, filepath))
    return '??'

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

def pp_cards(cards, fmt='@@'):
    s = ' '.join(chunk(cards, 2))
    s = fmt.replace('@@', s) 
    print(s)

SBAR_Y = 930
SBAR_X = [
    1860,
    1900,
    1960,
    2000,
    2040,
    2080,
    2120
]

def how_strong(image):
    strength = 1
    for x in SBAR_X:
        pixel = image.getpixel((x, SBAR_Y))
        # print(x, '->', pixel)
        if pixel[0] > 20:
            strength += 1
            continue
    return strength

XY_FOLD = (1000, 1000) # FOLD
XY_CALL = (1500, 1000) # CHECK, CALL
XY_BET = (2000, 1000) # BET, RAISE, ALL IN
XY_POT = (1700, 700) # POT bet
XY_PLUS = (1900, 700) # POT bet

def tap(xy):
    # fake_sleep(3)
    x, y = [str(n) for n in xy]
    _ = subprocess.check_output(['adb', 'shell', 'input', 'tap', x, y])
    time.sleep(.5) # 

def do_allin():
    tap(XY_BET)

def do_check():
    tap(XY_CALL)

def do_call():
    tap(XY_CALL)

def do_fold():
    tap(XY_FOLD)

def do_bet(size=0):
    tap(XY_BET)
    if size == 'POT':
        tap(XY_POT)
    elif size > 0:
        for _ in range(size // BIG_BLIND):
            tap(XY_PLUS)

    tap(XY_BET)

def do_raise(size):
    do_bet(size)

def fake_sleep(max_secs=3):
    time.sleep(random.uniform(0.8, max_secs))

def can_act(image):
    # one of three buttons already clicked
    if image.getpixel((1325, 1010))[0] > 200 or \
        image.getpixel((1755, 1010))[0] > 200 or \
        image.getpixel((895, 1010))[0] > 200:
        return False

    # third button is BET, RAISE or ALL-IN
    if can_bet(image) or can_raise(image):
        # debug
        save_image(image)
        return True

    return False

def can_bet(image):
    return image.getpixel((1910, 1010)) == (255, 255, 255, 255)

def can_raise(image): # only matches RAISE or ALL-IN (but not CALL ANY)
    return image.getpixel((1943, 1030)) == (255, 255, 255, 255)

def can_allin(image): # only matches ALL-IN (but not BET or RAISE)
    return image.getpixel((1955, 1015)) == (255, 255, 255, 255)

def can_check(image):
    return image.getpixel((1445, 1010)) == (255, 255, 255, 255)

# def can_call(image):
#     return not can_check(image) or can_allin(image)

def save_image(image):
    filepath = '/home/seb/screencaps-auto/%d.png' % time.time()
    if os.path.isfile(filepath):
        return
    image.save(filepath)

BIG_BLIND = 10000

def poll_table():

    while True:
        out = subprocess.check_output(['adb', 'exec-out', 'screencap', '-p'])
        image = Image.open(io.BytesIO(out))

        if not can_act(image):
            continue
        
        cards = read_mycards(image)
        if '??' in cards:
             continue

        score = how_strong(image)
        pp_cards(cards, '-> @@  %d/7' % score)

        board = read_board(image)
        if board:
            pp_cards(board)

        if score > 4:
            if can_allin(image):
                if score in (6, 7):
                    do_allin()
                else:
                    do_fold()
            else:
                if score == 5:
                    do_bet('POT')
                elif score == 6:
                    do_bet(BIG_BLIND * 5)
                elif score == 7:
                    do_bet(BIG_BLIND * 10)

        else:
            if can_check(image):
                do_check()
            elif can_allin(image):
                amount = read_mystack(image)
                if amount <= BIG_BLIND * score:
                    print('All in', amount)
                    do_allin()
                else:
                    do_fold()
            else: # CALL or RAISE
                amount = read_call(image)
                if amount <= BIG_BLIND * score-1:
                    print('Called', amount)
                    do_call()
                else:
                    do_fold()

        time.sleep(1) # 


REGION_MYSTACK_VALUE = (760, 915, 910, 945)
REGION_POT_VALUE = (1015, 590, 1150, 625)

from tesserocr import PyTessBaseAPI, PSM, OEM, image_to_text

def read_number(image, region):
    image = image.crop(region)
    image = image.convert('L')

    s = image_to_text(image, psm=7).rstrip() # PSM.SINGLE_BLOCK 6 PSM.SINGLE_LINE 7 PSM.SINGLE_WORD 8 PSM.SINGLE_CHAR 10

    # debug
    image.save('/tmp/value.png')
    print(s)

    if ' ' in s:
        _, s = s.split(' ')
    s = s.replace(',', '')
    
    if s.endswith('K'):
        s = s[:-1] + '000'
    elif s.endswith('M'):
        s = s[:-1] + '000000'
    return int(s)

def read_mystack(image):
    return read_number(image, REGION_MYSTACK_VALUE)

def read_pot(image):
    return read_number(image, REGION_POT_VALUE)

def read_call(image):
    first = 0
    last = 0
    for x in range(1350, 1700):
        pixel = image.getpixel((x, 990))
        if pixel == (255, 255, 255, 255):
            if not first:
                first = x
            last = x

    return read_number(image, (first-5, 975, last+5, 1045)) # REGION_CALL_VALUE


def test_call():
    d = '/home/seb/calls'
    for i, f in enumerate(sorted(os.listdir(d))):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)

        read_call(image)

def blah():
    d = '/home/seb/actions'
    # xy = (1885, 1010) # button 3
    # xy = (1905, 1010) # button 3
    xy = (1445, 1010) # button 2
    for i, f in enumerate(sorted(os.listdir(d))):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)
        pixel = image.getpixel(xy)
        print(pixel, image_path)
        # print('can_act:', can_act(image))
        print('can_bet:', can_bet(image))
        print('can_raise:', can_raise(image))
        print('can_allin:', can_allin(image))
        # print('can_check:', can_check(image))
        

if __name__ == '__main__':
    poll_table()

