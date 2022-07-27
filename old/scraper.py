import time
import os
import subprocess
import io
import random

from tesserocr import PyTessBaseAPI, PSM, OEM, image_to_text
from PIL import Image

import pickle
if os.path.isfile('VALUE_MAP.pkl'):
    with open('VALUE_MAP.pkl', 'rb') as f:
        VALUE_MAP = pickle.load(f)
        
else:
    VALUE_MAP = dict([
 ('73e3917d5fb0a5a9c323152c0af62830', '2'),
 ('6416ab035551ac0306795bfe55392899', '2'),
 ('83433264e725297484a0c00231ee109d', '2'),
 ('561e94cbca2dcb28b521942d76737f28', '2'),
 ('7115befdc69a0192ac8a0ec4c58e28d9', '2'),
 ('7ccf7c2358156150db6a5dcfdec0275a', '2'),
 ('0d6ae74f9913b203ea7985fc68d24c03', '2'),
 ('2ad06c7b7bab63535770f70487387dd0', '2'),
 ('a81209807cf3d57a39376c398fd2da3a', '3'),
 ('c171cb02532151ccf11dfd260a4cfb33', '3'),
 ('5c45237b221cf945765a9ae4147ba3f1', '3'),
 ('88461c6749b6cf490f1cd9d65f101691', '3'),
 ('8a2a6d479977d6731b235d97fabc911d', '3'),
 ('c89f4319e0aa6ca3d6101315d53bbd39', '3'),
 ('5daa72564afcc5bad812606653a1ccaa', '3'),
 ('50fc2d66576a0e15cf5103df16057326', '3'),
 ('21c91eb078155095026733e7d8315ab2', '4'),
 ('da777579e03556018c3895c8ea9066b8', '4'),
 ('6d23a1f31cbcee424518a87f71b3f584', '4'),
 ('51ba7bd668df7afb46b18eb6a83698ea', '4'),
 ('5825c17924eebcbb6671d942a5a3591d', '4'),
 ('6fc9194dfdc5d218f2c07dfbe503979b', '4'),
 ('647f9a48293fe23574a29abcbc885947', '4'),
 ('56299dbaa2acd651728d29c6d09b794c', '4'),
 ('722c34f620cd5d1e064edfc59a9bb70e', '5'),
 ('fc334385023275351ddd4321288ac8f0', '5'),
 ('8fe6ced07d7342c363d90461dc42890d', '5'),
 ('15111a36e783adce56ae5798ea3cb7b1', '5'),
 ('92fe7b47b1d649103cb9c12774c70ff1', '5'),
 ('66c6e72803465e8ba7766fcae7961a3f', '5'),
 ('8617e7a5d2f53c00e58e71eb7a6b47d5', '5'),
 ('b756294e25d9f343848cd1b4c5de9f29', '6'),
 ('0904cecd241c049033e0f77f3e1228b5', '6'),
 ('13eded4775eec081758454e1528e6ec4', '6'),
 ('47bed96ec083f28d90e9367f59dc8d23', '6'),
 ('8dce14389da8707d546301372b339e9d', '6'),
 ('c4afc7b3feb241d7905b53541f0e879d', '6'),
 ('4cb5b6083619fcc297199faa2774200c', '6'),
 ('6c7457119d47cb56bae3f474452856d4', '6'),
 ('0642c3a64d2d7d3bdda686d37573f006', '7'),
 ('8bdc32023a6d83f89801ca9b74a44f57', '7'),
 ('1c8562a56fe99cc3122dd3b42fe1f3ce', '7'),
 ('4b779715149bbb07d04df909cf548169', '7'),
 ('b6fe587832836beb3eec7e5469e63d10', '7'),
 ('a10a50a2325bffc38d544a04223e7f94', '7'),
 ('09cf9e30f12f716614d8493c91b46b20', '8'),
 ('db04bffa28bb1687f4371f5f0df08b7a', '8'),
 ('c2d89a6e6a2451b10162d553250f2d2e', '8'),
 ('248e49198895d424c379a80b6ea1e216', '8'),
 ('1f1cb4a388985b63616e43fe17feccff', '8'),
 ('c0712ec28e850049555290a6d788ed26', '8'),
 ('ea5588e1a8c7d973649fddba19373401', '8'),
 ('268b92ae8c16a5faaceb4a7bd10d1bff', '9'),
 ('d197ada2e2c8727cbcdc1442d5ddfbf5', '9'),
 ('ea80d6782b9267b618b0926337298e4d', '9'),
 ('f9c74f41be6da4ac11ef6c5802e9154a', '9'),
 ('378656f02bae1c0e214bdef04606bed1', '9'),
 ('009fb367cc1bb1f9fe280664abb2c8e4', '9'),
 ('3f8744c6ebd668893e135c792ec49103', '9'),
 ('a0531fae85d33e9b0557dd372932c291', '9'),
 ('04a8da7a5fe61208f8cbed2865a34319', '9'),
 ('3c69fdaabf1bd0c19a8085a69b65e657', 'A'),
 ('01517a90476426a6b0434e0a087e9f6e', 'A'),
 ('61440b8d8a07a6a8ea279f3388f60efb', 'A'),
 ('cf3ccd75e5dc235362e02bedfcb768d7', 'A'),
 ('64b1156bc06e412a066e0385ec848a22', 'A'),
 ('2b18c242c47fbeae0ad966698eb34c00', 'A'),
 ('1528e21003449dd8434a845051f82532', 'A'),
 ('65f244f8456fc99485786c1d34fdf75d', 'J'),
 ('d83dcea9be65fb2f02e216c06264cd29', 'J'),
 ('fb800280a7ab31d4e2c1d61e18bbbc35', 'J'),
 ('07ac6ebc41c027dbb4824e33883dc62e', 'J'),
 ('d674cbc79f42e342743f6c3b71db23ff', 'J'),
 ('5e2192f746c791bbc850843b9f48690c', 'J'),
 ('6a097b0be6e60ed1c1fd13219d009a96', 'J'),
 ('534ee3169ae6df8fe7688d66fe90ab0a', 'J'),
 ('de231d2a52351294865e32968a495c09', 'K'),
 ('afd8b3ae5c2d5007470f2edb0965075a', 'K'),
 ('d18823491aeed00e598acfc20c45d2dc', 'K'),
 ('02933681c1ae37b8817ec752d581a52a', 'K'),
 ('28751c1a5f8695556412d128fe409ab7', 'K'),
 ('b6c7fb9199afc969439d6b5a62786527', 'K'),
 ('a7036faa83f78b0807f425e308394570', 'K'),
 ('2d9ab057e3b7868dc9c065c052ac94d6', 'K'),
 ('796fefb0603a8c77fb175f31c3a7d956', 'Q'),
 ('aafe4475babb83d07736b3ea2ed77b69', 'Q'),
 ('4b0b33c8aee64ed945d4f0631ecab7d7', 'Q'),
 ('8e07aefb35fbf65a3054ec52ed34831c', 'Q'),
 ('17215766e62edc654754ca24c1e53225', 'Q'),
 ('25f2a391cf691641b98e8d27da7e9e5b', 'Q'),
 ('fe1a3b2cfa476c20284f316fb81fedba', 'Q'),
 ('1ec0962949413eb19197a3fe086640ed', 'Q'),
 ('1baf57637dbbf006a603fbba6ffa8559', 'Q'),
 ('75c1a32ea138f145c15f4000564f0d5d', 'T'),
 ('cddef64ec299f960898fcd7bdda42de1', 'T'),
 ('25e3a5f2da7866c67d8e426f36f36099', 'T'),
 ('775b03d33420ab9e925fa664373dd20b', 'T'),
 ('a7f47560f72bff18e737f66279c14dd5', 'T'),
 ('9f5081e3651a48f2b7a6edb1655fa012', 'T'),
 ('b36b3ae2dcb0f5dcd5bbfae57869e0f6', 'T'),
 ('a1ef5564ee398e251f0c65048a9aa55f', 'T'),
('caeeee6ee8961682374f16ddc574dd1d', 'c'),
('6be02ab5969d5fa865c25eb82f49553e', 'c'),
('d4a5b878ce845c2caad044c3e01c346b', 'd'),
('4cca33e659da1abb2bd379e555d3c4c6', 'd'),
('c905fb8ad7ba2e87c64455ce39d2295e', 's'),
('6d09fe331c523d140accda6b2029cf13', 's'),
('2081eab5e01fd653d7667426ec220722', 'h'),
('a64225e33fbe30a13e2258abd47e2925', 'h'),
('dc4fe8dede7f94bd9d360e5a4951eb68', 's'), # B1
('e6624e0f5747410a6aa8d1767180144a', 'd'), # B1
('dc4fe8dede7f94bd9d360e5a4951eb68', 's'), # B1
('e029b65a53e997bd0bdc20ff8fbb2874', 'c'), # B1
('514dae648a22818197ccbfacb6c2aafd', 'h'), # B1
('a3a95570801d3b966ac96cdb307aba3c', 's'), # B2
('b21ae73e481881058ce528d4872980d8', 'd'), # B2
('18744e55542ea379103aa2fd0a4b46a9', 'h'), # B2
('3fa1d928635cbe29d63fe6b0c8c51c8c', 'c'), # B2
('ff85c829aaa68b9c8fca9873a0f7bc43', 'c'), # B2 gold
('6a6e8166eba551a3b535e6041df1822f', 'd'), # B3
('e5d46718828c2f62cd40c20a804fc501', 's'), # B3
('265a6fe962d9562258b6eb2b25f89ba8', 'h'), # B3
('15329a71af1976f9b0223563fd424f50', 'c'), # B3
('07697f96a1cf675c2569111ec21ceff8', 'c'), # B3
('6c2c4196c9aaa51f4b3543437abdc250', 'd'), # B4
('09be9d1502a257b51b1f9780fdfc12fa', 'c'), # B4
('c697e50bca853a1402848d4607edee67', 's'), # B4
('0823b1fc6eb1ac60e86015ba979e8856', 'c'), # B4
('c697e50bca853a1402848d4607edee67', 'h'), # B4
('f53821d09f3d9def7e6d16a9b10549b5', 's'), # B5
('2b6d47a010c62418d548e371a5feaa44', 'c'), # B5
('7954d114ce9c36730ac165e1681e2fff', 'h'), # B5
('322296c00854bb1f6a83a7fab1c36c22', 'd'), # B5
    ])

# SUIT_MAP = {'caeeee6ee8961682374f16ddc574dd1d': 'c',
#             '6be02ab5969d5fa865c25eb82f49553e': 'c',
#             'd4a5b878ce845c2caad044c3e01c346b': 'd',
#             '4cca33e659da1abb2bd379e555d3c4c6': 'd',
#             'c905fb8ad7ba2e87c64455ce39d2295e': 's',
#             '6d09fe331c523d140accda6b2029cf13': 's',
#             '2081eab5e01fd653d7667426ec220722': 'h',
#             'a64225e33fbe30a13e2258abd47e2925': 'h'}

SCREENCAPS_MYCARDS = [('9s', '4h'), ('4d', 'Qh'), ('5d', '5c'), ('Qs', 'Th'), 
    ('4d', 'Tc'), ('Kc', '8c'), ('9h', '6d'), ('5h', 'Tc'), ('5h', 'Th'),
    ('Qd', '4c'), ('Js', '8s'), ('3c', '3d'), ('2h', '4s'), ('7s', '3s'), 
    ('3s', '7d'), ('5d', '7d'), ('Kh', '8s'), ('2h', '4d'), ('4d', 'Jh'), 
    ('9d', '4d'), ('Jc', 'Jh'), ('5h', 'Qs'), ('Th', 'As'), ('Ks', '4s'), 
    ('2s', 'Ks'), ('Qs', 'Ks'), ('2c', '4s'), ('3c', 'Ac'), ('3c', '4s'), 
    ('5h', 'Ts'), ('As', '2d'), ('Td', '8s'), ('8s', '5h'), ('Td', '9c'), 
    ('6h', '3d'), ('Qd', 'Jh'), ('9c', 'Ks'), ('Jh', 'As'), ('2s', 'Ad'), 
    ('3h', 'Kc'), ('Js', 'Kh'), ('8d', 'Ad'), ('Jc', '4c'), ('5h', '4d'), 
    ('7c', 'Js'), ('3c', '9d'), ('7d', 'Th'), ('7d', 'Qh'), ('6d', '8s'), 
    ('As', '8s'), ('5h', '6c'), ('6h', '9h'), ('2c', 'Jc'), ('Jc', 'Qh'), 
    ('2h', '6h'), ('6d', '7d'), ('8s', 'Jc'), ('Jh', '7h'), ('3h', '3s'), 
    ('6c', '6s'), ('4d', '4c'), ('4s', '9c'), ('Qc', '8s'), ('Kd', 'Tc'),
    ('4c', 'Ad'), ('3c', 'Kc'), ('Qd', '2c'), ('3h', 'Td'), ('5s', '3c'), 
    ('7d', '8h'), ('3c', 'Td'), ('Ad', '3s'), ('Qh', '7s'), ('Kh', 'Kc'), 
    ('2s', '4c'), ('Td', '2d'), ('2c', '9s'), ('Kd', '5h'), ('Ad', 'Jh'), 
    ('Qd', '2h'), ('As', 'Qh'), ('Ks', '3d'), ('Kd', 'Ks'), ('9h', '2h'), 
    ('Ks', 'Ah'), ('2s', '3h'), ('Th', '6c'), ('3h', 'Qc'), ('As', '2c'), 
    ('Ad', '9h'), ('9d', 'Tc'), ('7c', 'Qc'), ('8h', 'Ac'), ('3h', '7c'), 
    ('3h', 'Ah'), ('7c', 'As'), ('Jh', 'Js'), ('Ad', 'Ac'), ('Ks', '3h'), 
    ('5d', 'Th'), ('5c', '8s')]

SCREENCAPS_BOARD = [#('Qs', '8s', '8d'), 
    ('7d', '9s', '7s'), ('8s', '3s', 'Kh', '7d', 'Ks'),
    ('6c', '5s', '2h', '8c', '3c'), ('4c', 'Jd', 'Jc'), ('8d', 'Qh', 'Qc'), 
    ('Qh', '4h', '5h', '3c'), ('Qd', '5h', 'Kh', '2c'), ('Jc', '9h', 'Th'),
    ('2h', 'Jc', 'As', '3c', 'Th')]

def hash_image(image):
    from hashlib import md5
    return md5(image.tobytes()).hexdigest()

def build_maps(d='/home/seb/screencaps-pixel/'):
    value_map = {}
    # suit_map = {}

    for i, f in enumerate(sorted(os.listdir(d))):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)
        (card1, card2) = SCREENCAPS_MYCARDS[i]

        v1_hash = hash_image(image.crop(MYCARD1_VALUE_REGION))
        v2_hash = hash_image(image.crop(MYCARD2_VALUE_REGION))
        # s1_hash = hash_image(image.crop(MYCARD1_SUIT_REGION))
        # s2_hash = hash_image(image.crop(MYCARD2_SUIT_REGION))
        value_map[v1_hash] = card1[0]
        value_map[v2_hash] = card2[0]

    value_map = sorted(value_map.items(), key=lambda x: x[1])
    print('VALUE_MAP = %r' % value_map)
    return value_map


def scan_mycards(d='/home/seb/screencaps-pixel/'):
    for i, f in enumerate(sorted(os.listdir(d))):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)

        actual = read_mycards(image)
        pp_cards(actual)

        actual = actual[:2], actual[2:]
        expected = SCREENCAPS_MYCARDS[i]
        if actual != expected:
            print('actual: %s, expected: %s, image: %s' % (actual, expected, image_path))
            break
        # _ = input()

MYCARD1_VALUE_REGION = (1070, 755, 1135, 855)
MYCARD2_VALUE_REGION = (1245, 735, 1300, 840)
MYCARD1_SUIT_REGION = (1122, 870, 1170, 939)
MYCARD2_SUIT_REGION = (1232, 862, 1286, 925)
MYCARD_REGIONS = [
    MYCARD1_VALUE_REGION, 
    MYCARD1_SUIT_REGION,
    MYCARD2_VALUE_REGION,
    MYCARD2_SUIT_REGION
    ]

def lookup_suit(image, region):
    image = image.crop(region)
    image.save('/tmp/s.png')
    h = hash_image(image)
    # print(h)
    # return h
    # return SUIT_MAP.get(h, '?')
    return VALUE_MAP.get(h, '?')

def lookup_value(image, region):
    return VALUE_MAP.get(
        hash_image(image.crop(region)), '?')

import cv2
import numpy as np
def match_value(image, region):
    image = image.crop(region)
    image.save('/tmp/needle.png')

    img_rgb = cv2.imread('/tmp/needle.png')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    best_score = 0
    best_match = '?'
    for v in CARD_VALUES:
        template = cv2.imread('/home/seb/cv/%s.png' % v, 0)
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8)
        count = len(loc[0])+len(loc[1])
        if count > best_score:
            best_match = v
    print('is it?', best_match)

    input()

def register_value(image, region, prompt):
    while True:
        v = input(prompt)
        if not v:
            return '?'

        if v in CARD_VALUES:
            break

    h = hash_image(image.crop(region))
    VALUE_MAP[h] = v

    with open('VALUE_MAP.pkl', 'wb') as f:
        pickle.dump(VALUE_MAP, f)

    return v

def read_mycards(image):
    value1 = lookup_value(image, MYCARD_REGIONS[0])
    value2 = lookup_value(image, MYCARD_REGIONS[2])

    suit1 = lookup_suit(image, MYCARD_REGIONS[1])
    suit2 = lookup_suit(image, MYCARD_REGIONS[3])

    return value1+suit1+value2+suit2

BOARD1_VALUE_REGION = (800, 400, 850, 470)
BOARD2_VALUE_REGION = (920, 400, 970, 470)
BOARD3_VALUE_REGION = (1040, 400, 1090, 470)
BOARD4_VALUE_REGION = (1160, 400, 1210, 470)
BOARD5_VALUE_REGION = (1280, 400, 1130, 470)

BOARD1_SUIT_REGION = (802, 470, 835, 510)
BOARD2_SUIT_REGION = (921, 470, 954, 510)
BOARD3_SUIT_REGION = (1048, 470, 1080, 510)
BOARD4_SUIT_REGION = (1172, 470, 1205, 510)
BOARD5_SUIT_REGION = (1296, 470, 1329, 510)

def test_screencaps(d='/home/seb/screencaps-board/'):
    for i, f in enumerate(sorted(os.listdir(d))):
        image_path = os.path.join(d, f)
        image = Image.open(image_path)

        actual = read_board(image)
        
        # pp_cards(actual)
        
        # actual = tuple(c for c in actual if c in DECK)
        
        # expected = SCREENCAPS_BOARD[i]
        # if actual != expected:
        #     print('actual: %s, expected: %s, image: %s' % (actual, expected, image_path))
        #     break

        # _ = input()

def read_board(image):
    v1 = match_value(image, BOARD1_VALUE_REGION)
    v2 = match_value(image, BOARD2_VALUE_REGION)
    v3 = match_value(image, BOARD3_VALUE_REGION)
    v4 = match_value(image, BOARD4_VALUE_REGION)
    v5 = match_value(image, BOARD5_VALUE_REGION)

    # s1 = lookup_suit(image, BOARD1_SUIT_REGION)
    # s2 = lookup_suit(image, BOARD2_SUIT_REGION)
    # s3 = lookup_suit(image, BOARD3_SUIT_REGION)
    # s4 = lookup_suit(image, BOARD4_SUIT_REGION)
    # s5 = lookup_suit(image, BOARD5_SUIT_REGION)
    
    # board = v1+s1, v2+s2, v3+s3, v4+s4, v5+s5
    # board = tuple(c for c in board if c in DECK)

    return v1, v2, v3, v4, v5

def register_unknown(cards, image):
    while True:
        i = cards.find('?')
        if i == -1:
            return cards
        
        pp_cards(cards)
        v = register_value(image, MYCARD_REGIONS[i], '%s^ is: ' % (' ' * i))
        cards = cards.replace('?', v, 1)

CARD_VALUES = "23456789TJQKA"
CARD_SUITS = "shdc"
DECK = [x + y for x in CARD_VALUES for y in CARD_SUITS]

XY_FOLD = (1000, 1000)
XY_CALL = (1500, 1000)
XY_ANY = (2000, 1000)

def chunk(s, bs):
  return [s[i:i + bs] for i in range(0, len(s), bs)]

def pp_cards(cards, pfx=''):
    print(pfx, ' '.join(chunk(cards, 2)))

def tap(xy):
    x, y = [str(n) for n in xy]
    subprocess.check_output(['adb', 'shell', 'input', 'tap', x, y])

def poll_table():
    prev_cards = ''
    prev_board = ''

    while True:
        out = subprocess.check_output(['adb', 'exec-out', 'screencap', '-p'])
        image = Image.open(io.BytesIO(out))
        board = read_board(image)

        if prev_board != board: # if board and
            image.save('/home/seb/screencaps-auto/%d.png' % time.time())
            pp_cards(board, 'board')
            prev_board = board
            continue
        time.sleep(
            random.uniform(0.8, 2.2))
        tap(XY_FOLD)

        cards = read_mycards(image)
        if '??' in cards:
             continue
        
        # if '?' in cards:
        #     cards = register_unknown(cards, image)        

        if prev_cards != cards:
            pp_cards(cards, 'H')
            prev_cards = cards
            continue

        # tap(XY_CALL)

if __name__ == '__main__':
    poll_table()

#TODO
# always click on check/fold if hand strength shit

# board ocr
# def ocr_value(image, region):
#     image = image.crop(region)
#     image = image.convert('L')
#     t = image_to_text(image, psm=10).rstrip() # PSM.SINGLE_BLOCK 6 PSM.SINGLE_LINE 7 PSM.SINGLE_WORD 8 PSM.SINGLE_CHAR 10
#     if t.startswith('I'):
#         t = 'T'
#     return t

# def ocr_value(image, region, angle):
#     image = image.rotate(angle)
#     image = image.crop(region)

#     # image = image.convert('1')
#     # if angle > 0:
#     #     image.save('/tmp/r.png')
#     # else:
#     #     image.save('/tmp/l.png')

#     t = image_to_text(image, psm=8).rstrip() # PSM.SINGLE_BLOCK 6 PSM.SINGLE_LINE 7 PSM.SINGLE_WORD 8 PSM.SINGLE_CHAR 10
    
#     md5 = hash_image(image)
#     # print('raw:', t, 'md5:', md5)
#     # print('raw:', t)

#     if t.startswith('?'):
#         t = '2'
#     if t.startswith('ai') or t.startswith('i'):
#         t = '5'
#     if t.startswith('L'):
#         t = '7'
#     if t.startswith('g') or t.startswith('Q'):
#         t = '9'
#     if t.startswith('1'):
#         t = 'T'
#     if t.startswith('U') or t.startswith('0'):
#         t = 'Q'

#     t = ''.join(c for c in t if c in '123456789TJQKA')
#     return t
