from genericpath import isfile
import time
import os
import subprocess
import io
import random
import cv2
import numpy
import re
from tesserocr import image_to_text, PyTessBaseAPI, PSM
from hashlib import md5
from PIL import Image
from texasholdem import Card
from texasholdem.evaluator import  evaluate, rank_to_string, get_five_card_rank_percentage

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

def hash_image(image):
    return md5(image.tobytes()).hexdigest()

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

def pp_cards(cards, board, stats):
    s = ' '.join(re.sub('[\[\] ]', '', Card(c).pretty_string) for c in chunk(cards, 2))
    if board:
        s = s + '  %-14s' % ' '.join(re.sub('[\[\] ]', '', Card(c).pretty_string) for c in chunk(board, 2))
    else:
        s += ' ' * 16
    s = s + ' - ' + stats
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

def eval_rank(cards, board):
    if not board:
        return 0
    e = evaluate(
        cards = [Card(c) for c in chunk(cards, 2)],
        board = [Card(c) for c in chunk(board, 2)]
    )
    return int(get_five_card_rank_percentage(e) * 100)

XY_FOLD = (1000, 1000) # FOLD
XY_CALL = (1500, 1000) # CHECK, CALL
XY_BET = (2000, 1000) # BET, RAISE, ALL IN
XY_POT = (1700, 700) # POT bet
XY_PLUS = (1900, 700) # POT bet

def tap(xy):
    # fake_sleep(3)
    x, y = [str(n) for n in xy]
    _ = subprocess.check_output(['adb', 'shell', 'input', 'tap', x, y])
    time.sleep(.2) #

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

    # hand stength meter
    if image.getpixel((1760, 935)) != (132, 132, 138, 255):
        return False

    # third button is BET, RAISE or ALL-IN
    if can_bet(image) or can_raise(image):
        # debug
        save_image(image)
        return True

    return False

REGION_BUTTON3 = (1735, 960, 2145, 1059)
TEMPLATES_BUTTON3 = {}
for v in ['bet', 'raise', 'allin', 'callany']:
    tv = cv2.imread('./cv/buttons/%s.png' % v, 0)
    TEMPLATES_BUTTON3[v] = tv

def read_button3(image):
    return match_symbol(image, REGION_BUTTON3, TEMPLATES_BUTTON3)

def can_bet(image):
    return read_button3(image) == 'bet'

def can_raise(image):
    return read_button3(image) == 'raise'

def can_allin(image):
    return read_button3(image) == 'allin'

def can_check(image):
    return hash_image(image.crop((1440, 985, 1585, 1040))) == '5590b09a2a4c65a1e57aeb7a1d38d6b2'

# def can_call(image):
#     return not can_check(image) or can_allin(image)

def save_image(image):
    filepath = '/home/seb/screencaps-auto/%d.png' % time.time()
    if os.path.isfile(filepath):
        return
    image.save(filepath)

def poll_table():
    rank = 0
    while True:
        out = subprocess.check_output(['adb', 'exec-out', 'screencap', '-p'])
        image = Image.open(io.BytesIO(out))

        if not can_act(image):
            continue
        
        cards = read_mycards(image)
        if '??' in cards:
             continue

        board = read_board(image)
        score = how_strong(image)
        
        rank = eval_rank(cards, board)

        pp_cards(cards, board, '%d/8 %d%%' % (score, rank))

        if score > 4:
            if can_allin(image):
                if score >= 6:
                    do_allin()
                else:
                    do_fold()
            else:
                if score >= 6:
                    do_bet(BIG_BLIND * 10)
                else:
                    do_bet('POT')

        else:
            if can_check(image):
                do_check()
            elif can_allin(image):
                mystack = read_mystack(image)
                
                if mystack <= BIG_BLIND * score:
                    print('All in. %d <= BB*%d' % (mystack, score))
                    do_allin()
                else:
                    do_fold()
            else: # CALL or RAISE
                amount = read_call(image)
                if amount <= BIG_BLIND * (score-1):
                    print('Call. %d <= BB*%d' % (amount, score-1))
                    do_call()
                else:
                    do_fold()

        time.sleep(1) # 

def read_number(img, region, fiddle=False):
    for offset_x in range(4):
        for offset_y in range(4):
            r = (region[0]-offset_x, region[1]-offset_y, region[2]+offset_x, region[3]+offset_y)
            image = img.crop(r).convert('L')

            s = image_to_text(image, psm=7).rstrip() # PSM.SINGLE_BLOCK 6 PSM.SINGLE_LINE 7 PSM.SINGLE_WORD 8 PSM.SINGLE_CHAR 10
            # api = PyTessBaseAPI()
            # api.SetVariable("tessedit_char_whitelist", 'CALL 0123456789KM.,')
            # # api.SetPageSegMode(PSM.SINGLE_LINE)
            # api.SetImage(image)
            # s = api.GetUTF8Text()
            # debug
            # image.save('/tmp/value.png')
            # print('OCRed: %r' % s.strip())

            s = s.strip()
            if s.startswith('CALL '):
                s = s[5:]

            if re.match('[0-9,]+$', s):
                s = s.replace(',', '')
                return int(s)
            elif re.match('[0-9.]+[KMB]$', s):
                f = float(s[:-1])
                if s.endswith('K'):
                    f *= 1000
                elif s.endswith('M'):
                    f *= 1000000
                elif s.endswith('B'):
                    f *= 1000000000
                return int(f)
            elif not fiddle:
                break

    return None

def read_mystack(image, offset_x=-3, offset_x2=3):
    first = 0
    last = 0
    for x in range(765, 940):
        for y in range(912, 944):
            pixel = image.getpixel((x, y))
            if pixel[0] > 100:
                if first == 0:
                    first = x
                last = x
    return read_number(image, (first+offset_x, 913, last+offset_x2, 944), True) 
    # return read_number(image, (794, 913, 922, 944)) # n.mM OK /o/ y1:912-915 y2:940-944

def read_pot(image):
    return read_number(image, (1020, 590, 1170, 627))

def read_call(image):
    return read_number(image, (1350, 987, 1710, 1041), True)


# TESTS TESTS TESTS

EXPECTED_CALLS = [30000, 40000, 80000, 50000, 5000, 10000, 40000, 27260, 50450, 15700, 190000, 255000, 258230, 35000, 250000, 110000, 181830, 90000, 507000, 25000, 267850, 200000, 650000, 6160000, 100000, 1500000, 1000000, 200000, 600000]
EXPECTED_MYSTACKS = [160450, 125000, 85000, 75000, 60000, 55000, 45000, 424750, 417736, 367286, 349650, 286334, 813884, 1100000, 656930, 1100000, 1000000, 157169, 4800000, 2000000, 1300000, 3700000, 2900000, 7100000, 5100000, 3900000, 1500000, 6300000, 13800000]
EXPECTED_POTS = [50000, 40000, 160000, 40000, None, None, None, 25000, 30000, 130900, None, None, None, None, 450000, 120000, 40000, 180000, None, None, None, None, None, 750000, None, None, 5100000, None, 650000]
def test_ocr():
    dirpath = './tests/screencaps'
    for i, filename in enumerate(sorted(os.listdir(dirpath))):
        filepath = os.path.join(dirpath, filename)
        image = Image.open(filepath)

        actual = read_call(image)
        expected = EXPECTED_CALLS[i]
        if actual != expected:
            print(actual, '!=', expected)

        actual = read_mystack(image)
        expected = EXPECTED_MYSTACKS[i]
        if actual != expected:
            print(actual, '!=', expected)

        actual = read_pot(image)
        expected = EXPECTED_POTS[i]
        if actual != expected:
            print(actual, '!=', expected)

def blah():
    d = '/home/seb/actions-new'
    # xy = (1885, 1010) # button 3
    # xy = (1905, 1010) # button 3
    # xy = (1445, 1010) # button 2
    for i, f in enumerate(sorted(os.listdir(d))):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)
        # pixel = image.getpixel(xy)
        # print(pixel, image_path)
        print(image_path, can_bet(image))
        # print('can_act:', can_act(image))
        # print(
        # print('can_raise:', can_raise(image))
        # print('can_allin:', can_allin(image))
        # print('can_check:', can_check(image))

BIG_BLIND = 10000
# ANDROID_SERIAL=
if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 2:
        print('usage: <big blind>')
        exit(2)
    BIG_BLIND = int(argv[1])
    poll_table()

