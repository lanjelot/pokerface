import time
import os
import subprocess
import random
import re
import ast
import sys
from hashlib import md5
from PIL import Image
from select import select

import getch
import numpy
import cv2
from tesserocr import image_to_text, PyTessBaseAPI, PSM
from treys import Card
from treys import Evaluator


CARD_VALUES = "23456789TJQKA"
CARD_SUITS = "shdc"

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

CACHE_SYMBOL = {}
def match_symbol(image, region, templates):
    image = image.crop(region)
    # image.save('/tmp/needle.png')

    h = hash_image(image)
    if h in CACHE_SYMBOL:
        return CACHE_SYMBOL[h]

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

    CACHE_SYMBOL[h] = best_match
    return best_match

def read_mycards(image):
    v1 = match_symbol(image, REGION_MYCARD1_VALUE, TEMPLATES_MYCARD1_VALUE)
    v2 = match_symbol(image, REGION_MYCARD2_VALUE, TEMPLATES_MYCARD2_VALUE)

    s1 = match_symbol(image, REGION_MYCARD1_SUIT, TEMPLATES_MYCARD1_SUITS)
    s2 = match_symbol(image, REGION_MYCARD2_SUIT, TEMPLATES_MYCARD2_SUITS)

    return [v1+s1, v2+s2]

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
    return [c for c in board if c in VALID_CARDS]

def human_number(s):
    s = s.upper()
    f = float(s[:-1])
    if s.endswith('K'):
        f *= 1000
    elif s.endswith('M'):
        f *= 1000000
    elif s.endswith('B'):
        f *= 1000000000
    return int(f)

CACHE_NUMBER = {}
def read_number(image, region):
    image = image.crop(region)

    h = hash_image(image)
    if h in CACHE_NUMBER:
        return CACHE_NUMBER[h]

    image = image.convert('L')
    s = image_to_text(image, psm=7).rstrip() # PSM.SINGLE_BLOCK 6 PSM.SINGLE_LINE 7 PSM.SINGLE_WORD 8 PSM.SINGLE_CHAR 10

    # debug
    # image.save('/tmp/value.png')
    # print('OCRed: %r' % s.strip())

    s = s.strip()
    if s.startswith('CALL '):
        s = s[5:]

    n = None
    if re.match('[0-9,]+$', s):
        s = s.replace(',', '')
        n = int(s)
    elif re.match('[0-9.]+[KMB]$', s):
        n = human_number(s)

    if n is not None:
        CACHE_NUMBER[h] = n
    return n

XY_MY_BET = (1070, 655)
XY_OPP_BET = [
        (440, 620), # 1
        (455, 540), # 2
        (400, 325), # 3
        (635, 265), # 4
        (1415, 270),# 5
        (1665, 330),# 6
        (1600, 515),# 7
        (1600, 605),# 8
    ]

def read_pot(image):
    sx, sy = 1020, 590
    if image.getpixel((sx, sy)) != (0, 0, 0, 255):
        return None

    x1, y1 = 0, 0
    x2, y2 = 0, 0
    for x in range(sx, sx+150):
        for y in range(sy, sy+40):
            pixel = image.getpixel((x, y))
            if pixel[0] == 255:
                if x1 == 0:
                    x1 = x
                else:
                    x1 = min(x1, x)
                if y1 == 0:
                    y1 = y
                else:
                    y1 = min(y1, y)
                x2 = max(x2, x)
                y2 = max(y2, y)

    return read_number(image, (x1-5, y1-5, x2+5, y2+5))

def read_bet(image, xy=XY_MY_BET):
    sx, sy = xy
    if not (image.getpixel((sx, sy))[0] > 200 or image.getpixel((sx+150, sy))[0] > 200):
        return None

    x1, y1 = 0, 0
    x2, y2 = 0, 0
    for x in range(sx, sx+150):
        for y in range(sy, sy+40):
            pixel = image.getpixel((x, y))
            if pixel[0] == 0:
                if x1 == 0:
                    x1 = x
                else:
                    x1 = min(x1, x)
                if y1 == 0:
                    y1 = y
                else:
                    y1 = min(y1, y)
                x2 = max(x2, x)
                y2 = max(y2, y)

    return read_number(image, (x1-5, y1-5, x2+5, y2+5))

def read_bets(image):
    bets = {}
    for i, xy in enumerate(XY_OPP_BET):
        n = read_bet(image, xy)
        bets[i] = n

    return bets

def read_call(image):
    x1, y1 = 1710, 1045
    x2, y2 = 1350, 985
    for x in range(1350, 1710):
        for y in range(985, 1045):
            pixel = image.getpixel((x, y))
            if pixel[0] == 255:
                x1 = min(x1, x)
                y1 = min(y1, y)
                x2 = max(x2, x)
                y2 = max(y2, y)

    # for i in range(6):
    #     for j in range(6):
    #         print('ij: (%d, %d)' % (i, j))
    for i, j in [(0, 0), (0, 1), (0, 2), (0, 5), (0, 3), (1, 0), (1, 5)]:
        n = read_number(image, (x1-4-i, y1-5-j, x2+4+i, y2+5+j))
        if n is not None:
            return n

def read_mystack(image):
    for y1, y2 in [(0, 0), (0, 2), (0, 1), (0, 4), (0, 3), (1, 0), (2, 0), (2, 1), (1, 2), (2, 3), (1, 1), (-2, 2)]:
        n = read_number(image, (770, 912+y1, 950, 945-y2))
        if n is not None:
            break
    return n

REGION_BUTTON3 = (1735, 960, 2145, 1059)
TEMPLATES_BUTTON3 = {}
for v in ['bet', 'raise', 'allin', 'callany']:
    tv = cv2.imread('./cv/buttons/%s.png' % v, 0)
    TEMPLATES_BUTTON3[v] = tv

def read_button3(image):
    return match_symbol(image, REGION_BUTTON3, TEMPLATES_BUTTON3)

def save_image(image):
    # takes 1 second for .png extension, so only use for debugging or save as raw and convert later
    filepath = '/home/seb/screencaps-auto/%d.raw' % time.time()
    if os.path.isfile(filepath):
        return
    with open(filepath, 'wb') as f:
        f.write(image.tobytes())
    # image.save(filepath)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

class Timing:
  def __enter__(self):
    self.t1 = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.time = time.time() - self.t1

def chunk(s, bs):
  return [s[i:i + bs] for i in range(0, len(s), bs)]

def fake_sleep(max_secs=3):
    time.sleep(random.uniform(0.8, max_secs))

SPADE = "\u2660"
HEART = "\u2665"
DIAMOND = "\u2666"
CLUB = "\u2663"
RED = "\x1b[91m"
MAGENTA = "\x1b[95m"
BLUE = "\x1b[94m"
RESET = "\x1b[0m"

def color_cards(cards):
    colored = []
    for card in cards:
        value, suit = card[0], card[1]
        if suit == 's':
            colored.append(value + SPADE)
        elif suit == 'c':
            colored.append(BLUE+ value + CLUB + RESET)
        elif suit == 'h':
            colored.append(RED + value + HEART + RESET)
        elif suit == 'd':
            colored.append(MAGENTA + value + DIAMOND + RESET)
    return ' '.join(colored)

def lprint(s):
    print(s, end='', flush=True)


# clockwise starting from my left
VILLAINS = [
    (495, 709),
    (422, 450), # evade notifications poping up left
    (548, 260),
    (933, 263),
    (1351, 263),
    (1738, 260),
    (1773, 445),
    (1754, 696)
]

def read_villains(image):
    image = image.convert('L')
    villains = {}
    for i, (x, y) in enumerate(VILLAINS):
        p1 = image.getpixel((x, y))
        p2 = image.getpixel((x+1, y))
        # print(i, x, y, '->', p1, p2)
        if p1 - p2 > 20:
            villains[i] = 1

    return villains

XY_FOLD = (1000, 1000) # FOLD
XY_CALL = (1500, 1000) # CHECK, CALL
XY_BET = (2000, 1000) # BET, RAISE, ALL IN
XY_POT = (1700, 700) # POT bet
XY_HALF = (1700, 800) # 1/2 POT bet
XY_PLUS = (1900, 700) # POT bet
XY_ALL = (2070, 35) # ALL

def tap(xy):
    x, y = [str(n) for n in xy]
    _ = subprocess.check_output(['adb', 'shell', 'input', 'tap', x, y])
    time.sleep(.25)

def do_allin():
    lprint('All in')
    tap(XY_BET)

def do_check():
    lprint('Check')
    tap(XY_CALL)

def do_call():
    lprint('Call')
    tap(XY_CALL)

def do_fold():
    lprint('Fold')
    tap(XY_FOLD)

def do_bet(size=0):
    lprint('Bet ')
    tap(XY_BET)

    if size == 'POT':
        lprint(f'pot')
        tap(XY_POT)

    elif size == 'HALF':
        lprint(f'half')
        tap(XY_HALF)

    elif size == 'ALL':
        lprint(f'all in')
        tap(XY_ALL)

    elif size > 0:
        lprint(f'{size}')
        for _ in range(1, size):
            tap(XY_PLUS)

    tap(XY_BET)

def can_act(image):
    # one of three buttons already clicked
    if image.getpixel((1325, 1010))[0] > 200 or \
        image.getpixel((1755, 1010))[0] > 200 or \
        image.getpixel((895, 1010))[0] > 200:
        return False

    # third button is BET, RAISE or ALL-IN
    if can_bet(image) or can_raise(image) or can_allin(image):
        save_image(image) # debug
        return True

    return False

def can_bet(image):
    return read_button3(image) == 'bet'

def can_raise(image):
    return read_button3(image) == 'raise'

def can_allin(image):
    return read_button3(image) == 'allin'

def can_callany(image):
    return read_button3(image) == 'callany'

def can_call(image):
    return not can_check(image) and not btn2_disabled(image)

def can_check(image):
    return hash_image(image.crop((1440, 985, 1585, 1040))) == '5590b09a2a4c65a1e57aeb7a1d38d6b2'

def btn2_disabled(image):
    return hash_image(image.crop((1440, 985, 1585, 1040))) == '92653fa84141bd525b8e50c94d868abd'

def btns_disabled(image):
    return hash_image(image.crop((900, 985, 2100, 1040))) == '1dda2358e9ddbba55aa5f6742bf7dc82'

def read_button2(image):
    if can_check(image):
        return 'check'
    elif not btn2_disabled(image):
        return 'call'
    else:
        return 'only all in'

def what_stage(board):
    if len(board) == 0:
        return 'preflop'
    elif len(board) == 3:
        return 'flop'
    elif len(board) == 4:
        return 'turn'
    elif len(board) == 5:
        return 'river'
    return None

with open("preflop-odds.txt","r") as f:
    PREFLOP_ODDS = ast.literal_eval(f.read())

def preflop_odds(cards):
    c1, c2 = [Card.int_to_str(c) for c in
        sorted([Card.new(c) for c in cards], reverse=True)]
    s = c1[0] + c2[0]
    if c1[1] == c2[1]:
        s += 's'
    elif c1[0] != c2[0]:
        s += 'o'
    return PREFLOP_ODDS[s]

VALID_CARDS = [x + y for x in CARD_VALUES for y in CARD_SUITS]
FULL_DECK = [Card.new(c) for c in VALID_CARDS]

def montecarlo_odds(cards, board, num_villains):
    '''Compute win % using a Monte Carlo simulation'''
    board = [Card.new(c) for c in board]
    cards = [Card.new(c) for c in cards]

    all_cards = FULL_DECK[:]
    for c in cards + board:
        all_cards.remove(c)

    evaluator = Evaluator()

    rounds = 10000
    wins = 0
    for _ in range(rounds):
        deck = all_cards[:]
        random.shuffle(deck)

        full_board = board[:]
        for _ in range(5 - len(board)):
            full_board.append(deck.pop())

        my_score = evaluator.evaluate(full_board, cards)

        for _ in range(num_villains):
            their_cards = [deck.pop(), deck.pop()]
            their_score = evaluator.evaluate(full_board, their_cards)

            if my_score > their_score:  # they win
                break
        else:
            wins += 1

    return round(100 * wins / rounds, 2)

RANK_CLASS_TO_STRING = {
    0: "Royal",
    1: "St Fl",
    2: "Four",
    3: "Full",
    4: "Flush",
    5: "Strai",
    6: "Three",
    7: "Two P",
    8: "Pair",
    9: "High"
}

CACHE_ODDS = {}
def show_odds(cards, board, villains):
    s = color_cards(cards)
    s += ' ' * 2
    s += color_cards(board)
    if board:
        s += ' ' * 3*(5-len(board))
    else:
        s += ' ' * 14
    lprint(s)

    if board:
        evaluator = Evaluator()
        class_int = evaluator.get_rank_class(
            evaluator.evaluate(
                [Card.new(c) for c in cards],
                [Card.new(c) for c in board]))
        rank = RANK_CLASS_TO_STRING[class_int]
        lprint(f' {rank:>5}, ')
    else:
        lprint(' '*8)

    num_villains = len(villains)
    lprint(f'vs {num_villains}, ')

    if not board:
        odds = preflop_odds(cards)[num_villains+1]
    else:
        key = ''.join(cards) + ''.join(board) + str(num_villains)
        if key in CACHE_ODDS:
            odds = CACHE_ODDS[key]
        else:
            odds = CACHE_ODDS[key] = montecarlo_odds(cards, board, num_villains)

    lprint(f'win {odds:5.2f} ')
    return odds

def read_command():
    global MANUAL_MODE, BIG_BLIND
    i, _, _ = select([sys.stdin], [], [], .1)
    if i:
        s = i[0].readline().strip()
        if s == 'm': # mode switch
            MANUAL_MODE = not MANUAL_MODE
            print('Mode:', 'MANUAL' if MANUAL_MODE else 'AUTO')
        elif s.startswith('b'):
            BIG_BLIND = human_number(s.split()[-1])
            print('BIG_BLIND:', BIG_BLIND)

def read_choice(should_call=None):
    s = getch.getch()
    if s.isnumeric():
        do_bet(int(s))
    elif s == 'a':
        do_bet('ALL')
    elif s == 'h':
        do_bet('HALF')
    elif s == 'p':
        do_bet('POT')
    elif s == 'f':
        do_fold()
    elif s == 'c':
        if should_call is None:
            do_check()
    else:
        if should_call is None:
            do_check()
        elif should_call:
            do_call()
        else:
            do_fold()

def bet_or_check(win_odds, stage):
    if MANUAL_MODE:
        return read_choice()

    if stage == 'flop':
        if win_odds > 70:
            do_bet(2)
        elif win_odds > 50:
            do_bet(1)
        else:
            do_check()

    elif stage == 'turn':
        if win_odds > 80:
            do_bet('HALF')
        elif win_odds > 70:
            do_bet(2)
        elif win_odds >= 50:
            do_bet(1)
        else:
            do_check()

    elif stage == 'river':
        if win_odds > 90:
            do_bet('POT')
        elif win_odds > 80:
            do_bet('HALF')
        elif win_odds > 70:
            do_bet(2)
        elif win_odds >= 50:
            do_bet(1)
        else:
            do_check()
    else:
        do_check()

def call_or_fold(win_odds, stage, villains, image):

    call_ok = can_call(image)
    if call_ok:
        call_size = read_call(image)
    else:
        call_size = read_mystack(image)

    pot_size = read_pot(image) or 0

    if call_size <= BIG_BLIND:
        # assume everyone will call the BB
        pot_size += BIG_BLIND * (len(villains) + 1)
    else:
        # assume everyone after me will fold -> higher pot odds
        pot_size += sum(filter(None, read_bets(image).values()))
        pot_size += read_bet(image, XY_MY_BET) or 0
        pot_size += call_size

    pot_odds = threshold = 100 * call_size / pot_size

    if call_size > BIG_BLIND*2 and stage == 'preflop':
        stack_size = read_mystack(image)
        stack_odds = 100 * call_size / stack_size
        threshold = max(stack_odds, pot_odds)

    call_size = human_format(call_size)
    pot_size = human_format(pot_size)

    should_call = win_odds > threshold

    if should_call:
        lprint(f'> {threshold:5.2f} {call_size}/{pot_size}, ')
    else:
        lprint(f'< {threshold:5.2f} {call_size}/{pot_size}, ')

    if MANUAL_MODE:
        return read_choice(should_call)

    if should_call:
        if not call_ok: # can only tap "All-in"
            do_allin()
        elif stage in ('turn', 'river') and win_odds > 95:
            do_bet('ALL')
        else:
            do_call()
    else:
        do_fold()

def loop():
    while True:
        try:
            out = subprocess.check_output(['adb', 'exec-out', 'screencap'], timeout=1)
        except subprocess.TimeoutExpired:
            continue
        try:
            image = Image.frombytes('RGBA', (2340, 1080), out[16:])
        except ValueError:
            continue

        if btns_disabled(image):
            read_command()
            continue

        cards = read_mycards(image)
        if not all(c in VALID_CARDS for c in cards):
            continue

        villains = read_villains(image)
        if not len(villains) > 0:
            continue

        board = read_board(image)
        stage = what_stage(board)
        if not stage:
            continue

        if not can_act(image):
            read_bets(image)
            read_mystack(image)
            read_pot(image)
            continue

        win_odds = show_odds(cards, board, villains)

        if can_check(image):
            bet_or_check(win_odds, stage)
        else:
            call_or_fold(win_odds, stage, villains, image)

        print()
        time.sleep(1) #


def play():
    try:
        print('BIG_BLIND:', human_format(BIG_BLIND))
        print('MANUAL_MODE:', MANUAL_MODE)
        loop()
    except KeyboardInterrupt:
        pass

BIG_BLIND = 50000
MANUAL_MODE = True

if __name__ == '__main__':
    # test_perf()
    play()
    # if len(sys.argv) != 2:
    #     print('usage: <big blind>')
    #     exit(2)
    # BIG_BLIND = int(sys.argv[1])
    # play()




# TESTS TESTS TESTS

def test_perf():
    dirpath = '/home/seb/screencaps-auto'
    stats = []
    filemap = {}
    for i, filename in enumerate(sorted(os.listdir(dirpath))):
        filepath = os.path.join(dirpath, filename)
        image = Image.open(filepath)

        print(filepath)
        with Timing() as timing:
            # _ = read_bets(image)
            # _ = read_bet(image)
            # _ = read_pot(image)
            # n = read_mystack(image)
            if not read_button2(image) == 'call':
                continue
            n = read_call(image)
            if n is None:
                break

        # print(filepath, '%.2f' % timing.time)
        filemap[filepath] = timing.time
        stats.append(timing.time)
    print('max: %.2f' %  max(stats))
    print('min: %.2f' %  min(stats))
    print('avg: %.2f' %  (sum(stats)/len(stats)))
    return stats, filemap

EXPECTED_CALLS = [30000, 40000, 80000, 50000, 5000, 10000, 40000, 27260, 50450, 15700, 190000, 255000, 258230, 35000, 250000, 110000, 181830, 90000, 507000, 25000, 267850, 200000, 650000, 6160000, 100000, 1500000, 1000000, 200000, 600000, 600000, 500000]
EXPECTED_MYSTACKS = [160450, 125000, 85000, 75000, 60000, 55000, 45000, 424750, 417736, 367286, 349650, 286334, 813884, 1100000, 656930, 1100000, 1000000, 157169, 4800000, 2000000, 1300000, 3700000, 2900000, 7100000, 5100000, 3900000, 1500000, 6300000, 13800000, 4400000, 3700000]
EXPECTED_POTS = [50000, 40000, 160000, 40000, None, None, None, 25000, 30000, 130900, None, None, None, None, 450000, 120000, 40000, 180000, None, None, None, None, None, 750000, None, None, 5100000, None, 650000, 400000, 1700000]
EXPECTED_BETS = [None, None, None, None, 5000, None, 10000, None, None, None, 10000, 5000, 45000, 45000, None, None, None, None, 50000, 25000, 650000, None, 850000, None, 100000, 100000, None, 200000, None, 50000, 50000]
EXPECTED_BETS_SUM = [30000, 80000, 160000, 100000, 30000, 15000, 100000, 27260, 50450, 15700, 356730, 300000, 363230, 170000, 250000, 110000, 181830, 90000, 682000, 150000, 1755350, 325000, 3690000, 6160000, 800000, 2000000, 1000000, 900000, 600000, 650000, 550000]

def test_ocr():
    dirpath = './tests/screencaps'
    for i, filename in enumerate(sorted(os.listdir(dirpath))):
        filepath = os.path.join(dirpath, filename)
        image = Image.open(filepath)
        print(filepath)

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

        actual = read_bet(image)
        expected = EXPECTED_BETS[i]
        if actual != expected:
            print(actual, '!=', expected)

        bets = read_bets(image)
        actual = sum(filter(None, bets.values()))
        expected = EXPECTED_BETS_SUM[i]
        if actual != expected:
            print(actual, '!=', expected)

        print('.')


EXPECTED_VILLAINS = [7, 7, 3, 3, 3]
def test_count_villains():
    dirpath = './tests/count_villains'

    for i, filename in enumerate(sorted(os.listdir(dirpath))):
        filepath = os.path.join(dirpath, filename)
        image = Image.open(filepath)
        actual = len(read_villains(image))
        expected = EXPECTED_VILLAINS[i]
        if actual != expected:
            print(actual, '!=', expected)
        print(filepath, actual)

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