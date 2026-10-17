import random


class MyRandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.4):
        self.probability = probability
        self.sl = sl
        self.sh = sh

    def __call__(self, img):
        number = random.uniform(0, 1)

        if number > self.probability:
            return img

        height = img.size()[1]
        width = img.size()[2]
        image_area = height * width

        chosen_fraction = random.uniform(self.sl, self.sh)
        rectangle_area = chosen_fraction * image_area

        # Next we will turn this area into a height and width.
        return img