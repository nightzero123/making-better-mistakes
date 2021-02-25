import os
import shutil
path_train = "/data/likai/competitions/inaturalist-2019-fgvc6/train"
path_val = "/data/likai/competitions/inaturalist-2019-fgvc6/val"
paths = [path_train, path_val]
for path in paths:
    files = os.listdir(path)
    for file in files:
        pathsrc = os.path.join(path, file)
        print(len(file))
        while(len(file) < 4):
            file = '0' + file
        file = 'nat' + file
        pathtar = os.path.join(path, file)
        os.rename(pathsrc, pathtar)


