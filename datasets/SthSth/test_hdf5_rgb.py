import io
import h5py
from PIL import Image

h5_path = "/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/SthSth/dataset.hdf5"

with h5py.File(h5_path, "r") as f:
    keys = list(f.keys())
    print("num videos:", len(keys))
    vid = keys[0]
    print("first video id:", vid)
    print("num frames:", len(f[vid]))

    arr = f[vid][0]
    print(type(arr), arr.shape, arr.dtype)

    img = Image.open(io.BytesIO(arr.tobytes()))
    print("image size:", img.size, "mode:", img.mode)
    img.save("debug_rgb.jpg")