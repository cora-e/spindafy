from PIL import Image, ImageChops, ImageDraw
from random import randint
import numpy as np

class SpindaConfig:
    sprite_base = Image.open("res/spinda_base.png")
    sprite_mask = Image.open("res/spinda_mask.png")
    spot_masks = [
        np.array(Image.open("res/spots/spot_1.png")),
        np.array(Image.open("res/spots/spot_2.png")),
        np.array(Image.open("res/spots/spot_3.png")),
        np.array(Image.open("res/spots/spot_4.png"))
    ]
    spot_offsets = [
        (8, 6),
        (32, 7),
        (14, 24),
        (26, 25)
    ]
    def __init__(self):
        self.spots = [
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0)
        ]

    def __str__(self):
        return f"<SpindaConfig> {self.spots}"
    
    @staticmethod
    def from_personality(pers):
        self = SpindaConfig()
        self.spots[0] = (pers & 0x0000000f, (pers & 0x000000f0) >> 4)
        self.spots[1] = ((pers & 0x00000f00) >> 8, (pers & 0x0000f000) >> 12)
        self.spots[2] = ((pers & 0x000f0000) >> 16, (pers & 0x00f00000) >> 20)
        self.spots[3] = ((pers & 0x0f000000) >> 24, (pers & 0xf0000000) >> 28)
        return self

    @staticmethod
    def from_spot_locs(spot_locs):
        spot_locs = np.clip(spot_locs, 0, 15)
        self = SpindaConfig()
        for i in range(4):
            self.spots[i] = (int(spot_locs[2*i]), int(spot_locs[2*i+1]))
        return self

    @staticmethod
    def random():
        return SpindaConfig.from_personality(randint(0, 0x100000000))

    def get_personality(self):
        pers = 0x00000000
        for i, spot in enumerate(self.spots):
            pers = pers | (spot[0] << i*8) | (spot[1] << i*8+4)
        return pers

    def is_spot_arr(self):
        ret = np.zeros_like(self.sprite_base)[:,:,0]
        # Make a mask where the spots will be
        for i in range(len(self.spots)):
            x = self.spot_offsets[i][0] + self.spots[i][0]
            y = self.spot_offsets[i][1] + self.spots[i][1]
            mask = self.spot_masks[i][:,:,3]
            ret[y:y+mask.shape[0], x:x+mask.shape[1]] += mask == 255
        return ret

    def render_pattern(self, only_pattern = False, crop = False):
        mask_arr = np.asarray(self.sprite_mask)
        img_arr = np.zeros_like(self.sprite_base)
        if only_pattern:
            # White spots on black bkg
            spots_arr = 255*np.ones_like(self.sprite_base)
            bkg_arr = np.zeros_like(self.sprite_base)
            bkg_arr[:,:,3] = 255
        else:
            # Pull spots from mask, bkg from base sprite
            spots_arr = mask_arr
            bkg_arr = np.asarray(self.sprite_base)
        spots = self.is_spot_arr()
        for i in range(4):
            img_arr[:,:,i] = np.where(np.logical_and(spots, mask_arr[:,:,3]), spots_arr[:,:,i], bkg_arr[:,:,i])

        img = Image.fromarray(img_arr)

        if crop: img = img.crop((17, 15, 52, 48))

        return img

    def get_difference(self, target):
        result = self.render_pattern(only_pattern=True, crop=True).convert("RGB")
        return np.sum(ImageChops.difference(target, result)) / 3

if __name__ == "__main__":
    spin = SpindaConfig.from_personality(0x000FF0FF)
    spin.render_pattern().show()
    #print(hex(spin.get_personality()))