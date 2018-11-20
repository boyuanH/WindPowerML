import os

rootdir = 'C:\\Work\\ZKLF\\windpower\\windpower\\LocalMP\\data'


def directintodir(path):
    if os.path.isdir(path):
        for dirnames in os.walk(path):
            for dirname in dirnames[1]:
                if dirname == 'detect':
                    modelwavfile = open(os.path.join(path, "targetwavfile.txt"), "w")
                    for filenames in os.walk(os.path.join(path, 'detect')):
                        for filename in filenames[2]:
                            relatepath = path[len(rootdir)-len(path):]
                            modelwavfile.write(path+'/detect/'+filename + '\n')
                    modelwavfile.close()
                elif dirname == 'train':
                    targetwavfile = open(os.path.join(path, "modelwavfile.txt"), "w")
                    for filenames in os.walk(os.path.join(path, 'train')):
                        for filename in filenames[2]:
                            relatepath = path[len(rootdir)-len(path):]
                            targetwavfile.write(path+'/train/'+filename + '\n')
                    targetwavfile.close()
                else:
                    deepdir = os.path.join(path, dirname)
                    directintodir(deepdir)


directintodir(rootdir)


