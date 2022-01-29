
from scipy.fft import fftfreq


class Debug:
    count_dict = {}
    def __init__(self, place="", freq=50):
        if place not in Debug.count_dict:
            Debug.count_dict[place] = 0
        else:
            Debug.count_dict[place] += 1
        self.count = Debug.count_dict[place]
        if self.count % freq == 0:
            self.embed = True
        else:
            self.embed = False

def get_embed(header=""):
    _heading_ = header
    _debug_ = Debug(_heading_)
    if _debug_.embed:
        from IPython import embed
        embed(header = f"{_heading_}_{_debug_.count}")