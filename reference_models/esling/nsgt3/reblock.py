# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import numpy as np


def reblock(sseq, blocksize, dtype=None, fulllast=True, padding=0, multichannel=False):
    block = None
    dt = None
    chns = None

    if multichannel:
        channelize = lambda s: s
        unchannelize = lambda s: s
    else:
        channelize = lambda s: (s,)
        unchannelize = lambda s: s[0]

    for si in sseq:
        # iterate through sequence of sequences

        si = channelize(si)

        while True:
            if block is None:
                if dt is None:
                    # output dtype still undefined
                    if dtype is None:
                        dt = type(si[0][0])  # take type is first input element
                    else:
                        dt = dtype
                chns = len(si)

                block = np.empty((chns, blocksize), dtype=dt)
                blockrem = block

            sout = [sj[: blockrem.shape[1]] for sj in si]
            avail = len(sout[0])
            for blr, souti in zip(blockrem, sout):
                blr[:avail] = souti  # copy data per channel
            si = [sj[avail:] for sj in si]  # move ahead in input block
            blockrem = blockrem[:, avail:]  # move ahead in output block

            if blockrem.shape[1] == 0:
                # output block is full
                yield unchannelize(block)
                block = None
            if len(si[0]) == 0:
                # input block is exhausted
                break

    if block is not None:
        if fulllast:
            blockrem[:] = padding  # zero padding
            ret = block
        else:
            # output only filled part
            ret = block[:, : -len(blockrem[0])]
        yield unchannelize(ret)
