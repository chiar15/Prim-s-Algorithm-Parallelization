import os
import shutil

os.chdir("Information") 

for i in range(4):
    opt="opt"+str(i)
    os.mkdir(opt)
    

os.chdir("../PrimResults") 

for i in range(4):
    opt="opt"+str(i)
    os.mkdir(opt)
