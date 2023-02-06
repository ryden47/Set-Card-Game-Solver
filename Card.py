from feature_detector import *

c2clr = {'0': "red   ", "1": "green ", "2": "purple"}
c2num = {'0': "1", "1": "2", "2": "3"}
c2shd = {'0': "empty", "1": "full ", "2": "half "}
c2shp = {'0': "꧰", "1": "♢", "2": " ~"}


class Card:
    def __init__(self, cropped_image, contour):
        self.image = cropped_image
        self.contour = contour
        self.shape_rgb = None
        self.shape_focus = None
        self.number = identify_number(self)
        self.shape = identify_shape(self)
        self.shading = identify_shading(self)
        self.color = identify_color(self)
        self.id = (self.color, self.number, self.shading, self.shape)  # '0000', '0001', '0002', '0010', ...

    def speak(self):
        print(f"{self.number} - {self.shape} - {self.shading} - {self.color}")
        return f"{self.number} - {self.shape} - {self.shading} - {self.color}"
