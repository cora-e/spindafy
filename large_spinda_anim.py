from glob import glob
from pathlib import Path
from argparse import ArgumentParser
from large_spinda import to_spindas
import numpy as np
import multiprocessing

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_directory")
    parser.add_argument("output_directory")
    parser.add_argument("skip", type=int, default=0, nargs="?")
    parser.add_argument("--skip-even", action="store_true")
    parser.add_argument("--skip-odd", action="store_true")

    args = parser.parse_args()

    # find all input images
    inputs = np.sort(glob(args.input_directory + "/*"))

    # create output directories if they don't exist
    Path(args.output_directory).mkdir(parents=True, exist_ok=True)
    Path(args.output_directory + "/pids").mkdir(parents=True, exist_ok=True)

    pool = multiprocessing.Pool(processes=cpus)

    for n, filename in enumerate(inputs):
        print(f"STARTING FRAME #{n:0>4}! — ({n/len(inputs) * 100}%)")

        if n < args.skip:
            print(f"skipping first {args.skip} frames!")
            continue

        if n%2 == 0 and args.skip_even:
            print(f"skipping even frames!")
            continue

        if n%2 != 0 and args.skip_odd:
            print(f"skipping odd frames!")
            continue

        if len(glob(args.output_directory + f"/frame{n:0>4}*")) > 0:
            print("frame already found! skipping.")
            continue

        (img, pids) = to_spindas(filename, pool)

        output_filename = args.output_directory + f"/frame{n:0>4}.png"
        img.save(output_filename)

        np.savetxt(args.output_directory + f"/pids/frame{n:0>4}.json", pids)