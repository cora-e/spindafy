from PIL import Image
import PIL.ImageOps
import json
import itertools

import numpy as np
from scipy.optimize import minimize

from spindafy import SpindaConfig
from spinda_optimizer import evolve

PREDEFINED = {
    "ALL_WHITE": 0x393d9888,
    "ALL_BLACK": 0xff200000
}
PREDEFINED_ARRS = {
    "ALL_WHITE": np.array([0x8, 0x8, 0x8, 0x9, 0xd, 0x3, 0x9, 0x3]),
    "ALL_BLACK": np.array([0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0xf, 0xf])
}

def spinda_loss(spot_locs, sub_target):
    """Loss function for a Spinda pattern with spots in `spot_locs`,
    vs equivalent-size target image `sub_target`.
    """
    return SpindaConfig.from_spot_locs(spot_locs).get_difference(sub_target)

def spinda_jacobian(spot_locs, sub_target):
    """Jacobian (dervatives against all parameters) for a Spinda pattern with
    spots in `spot_locs`, vs equivalent-size target image `sub_target`.
    These derivatives are taken almost exactly as `numpy` would do it automatically,
    except that they force a step size of 1 pixel.  This is, of course, since moving
    less than a pixel will not change the loss function, which is kind of the thing
    we want.
    """
    # The derivatives have to be this wide, the fn only changes on integers
    # unfortunately that makes the solver real feisty
    ret = np.zeros((len(spot_locs),))
    for loc in range(len(spot_locs)):
        new_locs = spot_locs.copy()
        new_locs[loc] = spot_locs[loc] + 1
        jhi = SpindaConfig.from_spot_locs(new_locs).get_difference(sub_target)
        new_locs[loc] = spot_locs[loc] - 1
        jlo = SpindaConfig.from_spot_locs(new_locs).get_difference(sub_target)
        ret[loc] = (jhi - jlo) / 2
    return ret

def find_best_sub_spinda(x, y, target):
    """Optimize spot layout at x,y subimage of a full mosaic.
    `target` is the full mosaic, so we don't have to pre-slice
    before applying this function in parallel.
    """
    sub_target = target.crop((
        x*25, y*20,
        x*25+35,
        y*20+33
    ))
    # Check for predefined spinda patterns!
    if np.all(np.greater_equal(sub_target, 128)):
        return SpindaConfig.from_personality(PREDEFINED["ALL_WHITE"])
    # CEP: It looked like the original version of this check *worked*,
    # but I didn't *understand* it, so I made it slower.
    elif np.all(np.less_equal(sub_target.getchannel(0), 127)) and \
         np.all(np.less_equal(sub_target.getchannel(1), 127)) and \
         np.all(np.less_equal(sub_target.getchannel(2), 127)):
        return SpindaConfig.from_personality(PREDEFINED["ALL_BLACK"])
    else:
        # Random seeds are visible due to the number of solver failures
        #seed_locs = np.random.randint(16, size=8)
        # Seed "white" to catch the annoying solver errors
        #seed_locs = PREDEFINED_ARRS["ALL_WHITE"]
        # Seed "black" to hide them
        seed_locs = PREDEFINED_ARRS["ALL_BLACK"]
        opt_pid = minimize(spinda_loss, seed_locs, args=(sub_target,),
                            jac=spinda_jacobian, bounds=[(0,15),]*8)
        return SpindaConfig.from_spot_locs(opt_pid.x)

def sub_spinda_loc(xy_tuple, target):
    """Wrapper for above. Unpack tuple, keep location in return so we can reconstruct
    the image from the list of these returns.
    """
    return (xy_tuple[0], xy_tuple[1], find_best_sub_spinda(xy_tuple[0], xy_tuple[1], target))

def to_spindas(filename, pool, invert = False):
    """Use a multiprocessing `pool` to Spindafy an image at `filename`"""
    with Image.open(filename) as target:
        target = target.convert("RGB")
        if invert: target = PIL.ImageOps.invert(target)

        num_x = int((target.size[0]+10)/25)
        num_y = int((target.size[1]+13)/20)

        print(f"Size: {num_x} * {num_y}")

        xy_range = itertools.product(range(num_x), range(num_y))
        #best_list = [(x, y, find_best_sub_spinda(x, y, target)) for x, y in xy_range]
        best_list = pool.starmap(sub_spinda_loc, zip(xy_range, itertools.repeat(target)))
        #best_list = [(x, y, evolve(target.crop((x*25, y*20,
        #                                        x*25+35,
        #                                        y*20+33
        #                                        )), 100, 10)[1]) for x, y in xy_range]

        # Consolidate
        img = Image.new("RGBA", (39 + num_x * 25, 44 + num_y * 20))
        pids = np.zeros((num_x, num_y), dtype=np.uint32)
        for x, y, spinda in best_list:
            spinmage = spinda.render_pattern()
            img.paste(
                spinmage,
                (x * 25, y * 20),
                spinmage
            )
            pids[x][y] = spinda.get_personality()

        return (img, pids)
    
if __name__ == "__main__":
    (img, pids) = to_spindas("doom/test.png", 100, 10)
    img.resize((img.size[0]*10, img.size[1]*10), Image.Resampling.NEAREST).show()
    img.save("doom/test_res.png")
    with open("doom/test.json", "w") as f:
        json.dump(pids, f)