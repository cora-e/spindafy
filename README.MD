# Spindafy
This is a simple script to find a personality value (PID) that most closely matches a target image when applied as a pattern to the Pokémon Spinda!

You should go see the [original](https://github.com/Ovidios/spindafy), this is just a tribute.  I was inspired by [adef's vid](https://www.youtube.com/watch?v=BuV_9XW8ymo), of course, and the spirit of procrastination.  All credit for the real work was them, I'm just playing with the result.

This version has been optimized for speed a bit, most prominently by replacing the cool genetic algorithm with a much less cool but ~60x faster call out to scipy's bounded `minimize` (L-BFGS-B I guess?).  I had to use a custom Jacobian since the loss function between original and "Spindafied" image is discrete (can't have a spot between pixels!).  I've learned that apparently optimizing a discrete function, even an approximant of a nice smooth one, sucks a lot! So the solver struggles: while the results are faster, they lack the same visual fideility ("Spidelity?") as the original.  Doesn't stop me from doing crimes:
![adef has an announcement](res/adef_announcement.gif)

I've also introduced some incompatibilities, or I'd just make this a PR (json -> numpy output of PIDs, removed some features, and I wreaked some true havoc on ``large_spinda.py``).  I'm happy to work to upstream this stuff if there's suddenly a community of like-minded Spinda-pattern sickos that would want an alternative "Spindafiability solver."

However, after actually learning how Spinda patterns work, I'm sure there are much more dramatic optimizations than this: don't bother solving at all, just find an approximate inverse function!  You could find particularly dark/light locations (maybe with edge detection?) and then translate their centers or borders into suitable spot locations, with the additional hard-coded "white" and "black" available for removing spots that aren't needed.  While technically the spots *can* overlap for some PIDs, they don't really need to in order to replicate (almost) any pixel configuration, so there should be a unique satisfactory configuration available just from arranging the spots to follow the clearest light/dark edges.

There's also a lot of room to optimize images for Spindafication beforehand: thresholding and compressing them more intelligently to 1-bit BW might help both the program speed and the visual readability of the final Spindafied version (not least by reducing solver failures, hopefully).

File list:
- ``spindafy.py`` defines the ``SpindaConfig`` class, representing a Spinda point configuration. It also contains a method to render the pattern to a PIL Image, and to compare it to a target image.
- ``large_spinda.py`` implements the main mosaic generator, splitting an image into zones and giving each a Spindafied replacement.
- ``large_spinda_anim.py`` makes an animation of ``large_spinda.py`` results.
- ``spinda_optimizer.py`` is the untouched original genetic algorithm version, for comparison.