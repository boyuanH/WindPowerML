import os

rootdir = 'C:\\tmp\\Audio\\32'
linux_root_dir = '/home/ubuntu/localMP/data/32'


def directintodir(path, linux_dir):
    if os.path.isdir(path):
        for dirnames in os.walk(path):
            for dirname in dirnames[1]:
                if dirname == 'Validate':
                    modelwavfile = open(os.path.join(path, "targetwavfile.txt"), "w")
                    for filenames in os.walk(os.path.join(path, 'Validate')):
                        for filename in filenames[2]:
                            relatepath = path[len(rootdir)-len(path):]
                            modelwavfile.write(linux_dir+'/Validate/'+filename + '\n')
                    modelwavfile.close()
                elif dirname == 'Train':
                    targetwavfile = open(os.path.join(path, "modelwavfile.txt"), "w")
                    for filenames in os.walk(os.path.join(path, 'train')):
                        for filename in filenames[2]:
                            relatepath = path[len(rootdir)-len(path):]
                            targetwavfile.write(linux_dir+'/Train/'+filename + '\n')
                    targetwavfile.close()
                else:
                    deepdir = os.path.join(path, dirname)
                    linux_deepdir = os.path.join(linux_dir, dirname)
                    directintodir(deepdir, linux_deepdir)


directintodir(rootdir, linux_root_dir)


