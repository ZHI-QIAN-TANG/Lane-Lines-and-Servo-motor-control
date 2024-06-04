import cv2
import typing

U = typing.TypeVar("U")
context = typing.Callable[[], U]
dispatch = typing.Callable[[U], typing.NoReturn]
def state[T](value: T) -> tuple[context[T], dispatch[T]]:
    val = value
    
    def getValue():
        return val
    
    def setValue(v: T):
        nonlocal val
        val = v
        
    return (getValue, setValue)

    
def maskTrack(win: str) -> dict[str, context]:
    lowh = state(11)
    highh = state(34)
    lows = state(5)
    highs = state(80)
    lowv = state(198)
    highv = state(255)
    
    # lowh = state(0)
    # highh = state(180)
    # lows = state(0)
    # highs = state(255)
    # lowv = state(0)
    # highv = state(255)
    
    
    cv2.createTrackbar("low-h", win, lowh[0](), 180, lowh[1])
    cv2.createTrackbar("high-h", win, highh[0](), 180, highh[1])
    cv2.createTrackbar("low-s", win, lows[0](), 255, lows[1])
    cv2.createTrackbar("high-s", win, highs[0](), 255, highs[1])
    cv2.createTrackbar("low-v", win, lowv[0](), 255, lowv[1])
    cv2.createTrackbar("high-v", win, highv[0](), 255, highv[1])
    
    return {
        "lowh": lowh[0],
        "lowv": lowv[0],
        "lows": lows[0],
        "highh": highh[0],
        "highv": highv[0],
        "highs": highs[0]
    }
    
def cannyTrack(win: str):
    low = state(50)
    high = state(80)
    
    cv2.createTrackbar("low", win, low[0](), 255, low[1])
    cv2.createTrackbar("high", win, high[0](), 255, high[1])
    
    return {"low": low[0], "high": high[0]}