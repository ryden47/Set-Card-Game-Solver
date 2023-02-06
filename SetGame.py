from feature_detector import *
from GUI_utils import showone
from IMG_manipulations import rotate_img
from Card import Card
from itertools import combinations


class Game:
    def __init__(self, filename, num_of_cards=12):
        self.filename = filename.split('/')[-1][:-4]
        self.gray_image = cv2.imread(filename, 0)
        self.origin_image = cv2.imread(filename, 1)
        self.num_of_cards = num_of_cards
        self.cards = []

    def identify_cards(self):
        card_images = find_contours(self)
        rotated_card_images = [(rotate_img(img[0]), img[1]) for img in card_images]
        self.cards = [Card(card, cnt) for card, cnt in rotated_card_images]

    def find_sets(self):
        showone(self.origin_image, scale=0.2)
        for card1, card2, card3 in combinations(self.cards, r=3):
            if self.is_set(card1, card2, card3):
                self.display_triplets(card1, card2, card3)

    def display_triplets(self, card1, card2, card3):
        image = self.origin_image.copy()
        cv2.drawContours(image, [card1.contour], -1, (0, 186, 50), thickness=40, lineType=cv2.LINE_AA)
        cv2.drawContours(image, [card2.contour], -1, (0, 186, 50), thickness=40, lineType=cv2.LINE_AA)
        cv2.drawContours(image, [card3.contour], -1, (0, 186, 50), thickness=40, lineType=cv2.LINE_AA)
        showone(image, scale=0.2, save2path=f"./output/{input('name?')}.jpg")

    def is_set(self, card1, card2, card3):
        shades = len({card1.shading, card2.shading, card3.shading})
        number = len({card1.number, card2.number, card3.number})
        colors = len({card1.color, card2.color, card3.color})
        shapes = len({card1.shape, card2.shape, card3.shape})
        if shades == 2 or number == 2 or colors == 2 or shapes == 2:
            return False
        return True




