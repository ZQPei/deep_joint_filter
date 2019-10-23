import datetime
import dominate
from dominate.tags import *
import os
import glob
import numpy as np


class HTML(object):
    def __init__(self, html_title):
        self.doc = dominate.document(title=html_title)
        self.doc.add(h1(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))


    def display(self, path, flist, nrow=4, width=256):
        '''
            for a single image dir
        '''
        flist = self.load_flist(flist)
        basename_list = [os.path.basename(fn) for fn in flist]
        n = len(basename_list)
        idx = 0

        while idx < n:
            self.add_table()
            with self.t:
                with tr():
                    for _ in range(nrow):
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                basename = basename_list[idx]
                                linker = os.path.join(path, basename)
                                with a(href=linker):
                                    img(style="width:%dpx" %(width), src=linker)
                                    idx += 1
                                br()
                                p(basename)


    def compare(self, flist, width=256, **path_ext_dict):
        '''
            for multiple image dirs with same image fname list
        '''
        flist = self.load_flist(flist)
        basename_list = [os.path.basename(fn) for fn in flist]

        for basename in basename_list:
            self.add_header(basename)
            self.add_table()
            with self.t:
                with tr():
                    for txt, path_ext in path_ext_dict.items():
                        path, ext = path_ext
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=os.path.join(path, basename)):
                                    img(style="width:%dpx" %(width), src=os.path.join(path, os.path.splitext(basename)[0]+ext))
                                br()
                                p(str(txt))


    def add_header(self, string):
        self.doc.add(h3(string))

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def save(self, fn):
        with open(fn, 'w') as foo:
            foo.write(self.doc.render())


    @staticmethod
    def load_flist(flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, np.ndarray):
            return flist.tolist()

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    flist = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                    if flist.ndim > 1:
                        flist = flist[:,0]
                    return flist.tolist()
                except:
                    return [flist]

        return []


if __name__ == "__main__":
    # html = HTML("html_test")
    # path = 'results_120000'
    flist = "./checkpoints/places2_dfg_bs1_size256_sblk11111_0602_wclsinfo_nomultidis_novgggrad/results_120000"
    # html.display(path, flist)
    # savename = "./checkpoints/places2_dfg_bs1_size256_sblk11111_0602_wclsinfo_nomultidis_novgggrad/html_test.html"
    # html.save(savename)

    html_compare = HTML("html_compare_test")
    path_ext_dict = {
        'original': ['original', '.jpg'],
        'round_40000': ['results_40000', '.jpg'],
        'round_120000': ['results_120000', '.jpg'],
        'round_360000': ['results_360000', '.jpg'],
    }
    html_compare.compare(flist, **path_ext_dict)
    savename = "./checkpoints/places2_dfg_bs1_size256_sblk11111_0602_wclsinfo_nomultidis_novgggrad/html_test_compare.html"
    html_compare.save(savename)
    import ipdb; ipdb.set_trace()