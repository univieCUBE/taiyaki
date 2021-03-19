from typing import Final

from taiyaki.activation import swish
from taiyaki.layers import (
    Convolution,
    Lstm,
    Reverse,
    Serial,
    GlobalNormFlipFlopCatMod,
    ModifiedBasesClassifier,
)


def network(insize=1, size=256, winlen=19, stride=5, alphabet_info=None):
    winlen2: Final[int] = 5
    pooling_strategy: Final[str] = "avg"  # XXX Make this available in the
    pool_size: Final[int] = 1_000         #     function signature?

    return Serial([
        Convolution(insize, 4, winlen2, stride=1, fun=swish),
        Convolution(4, 16, winlen2, stride=1, fun=swish),
        Convolution(16, size, winlen, stride=stride, fun=swish),
        Reverse(Lstm(size, size)),
        Lstm(size, size),
        Reverse(Lstm(size, size)),
        Lstm(size, size),
        Reverse(Lstm(size, size)),
        GlobalNormFlipFlopCatMod(size, alphabet_info),
        ModifiedBasesClassifier(
            alphabet_info=alphabet_info,
            pooling_strategy=pooling_strategy,
            pool_size=pool_size,
        ),
    ])
