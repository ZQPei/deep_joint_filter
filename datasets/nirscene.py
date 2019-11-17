import os
import random
import PIL.Image as Image
from scipy.misc import imread, imsave


from gauss_noise import add_gaussian_noise


def is_image(fn):
    return True if os.path.splitext(fn)[1] in ['.jpg', '.jpeg', '.png', '.bmp', ".tiff"] else False


def load_flist(path):
    assert isinstance(path, str) and os.path.isdir(path)

    walk = os.walk(path)
    flist = []
    for parentname, _, filelist in walk:
        flist += [os.path.join(parentname, fn) for fn in filelist if is_image(fn)]
    
    flist.sort()
    return flist


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def resave(ori_path, tgt_path):
    flist = load_flist(ori_path)
    flist_rgb = list(filter(lambda s: "_rgb" in s, flist))
    flist_nir = list(filter(lambda s: "_nir" in s, flist))

    flist_rgb.sort()
    flist_nir.sort()


    total = len(flist_rgb)
    noise_sigma = 25
    index_trainset = random.sample(range(total), int(total*0.8))
    index_testset = [x for x in range(total) if x not in index_trainset]
    index_trainset.sort()
    index_testset.sort()
    index_dict = {
        "train": index_trainset,
        "test": index_testset
    }


    for set in ["train", "test"]:
        tgt_rgb_path = os.path.join(tgt_path, set, "rgb")
        tgt_nir_path = os.path.join(tgt_path, set, "nir")
        tgt_noise_path = os.path.join(tgt_path, set, "noise")
        create_dir(tgt_rgb_path)
        create_dir(tgt_nir_path)
        create_dir(tgt_noise_path)


        for i in index_dict[set]:
            print("\rProgress: %d/%d"%(i, total), end='')
            # rgb
            fn_rgb = flist_rgb[i]
            foldername = os.path.basename(os.path.dirname(fn_rgb))
            basename = os.path.splitext(os.path.basename(fn_rgb))[0]
            fn_rgb_tgt = os.path.join(tgt_rgb_path, foldername + "_" + basename + ".png")
            im = imread(fn_rgb)
            imsave(fn_rgb_tgt, im)

            # add noise
            fn_noise_tgt = os.path.join(tgt_noise_path, foldername + "_" + basename.replace("rgb", "noise") + ".png")
            im_noise = add_gaussian_noise(im, noise_sigma)
            imsave(fn_noise_tgt, im_noise)

            # nir
            fn_nir = flist_nir[i]
            foldername = os.path.basename(os.path.dirname(fn_nir))
            basename = os.path.splitext(os.path.basename(fn_nir))[0]
            fn_nir_tgt = os.path.join(tgt_nir_path, foldername + "_" + basename + ".png")
            im = imread(fn_nir)
            imsave(fn_nir_tgt, im)

    print("done")






if __name__ == "__main__":
    resave("/data/pzq/RGB-NIR/nirscene1", "/data/pzq/RGB-NIR/tiff2png")


