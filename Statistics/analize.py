import os
import shutil

def analizeFile(sourceFile,resultFile):#analize the source file and save the information in the result file
    f=open(sourceFile,"r")#open file and read all its lines
    lines=f.readlines()
    data={}
    num={}
    for line in lines:#for each line get the informations
        type,mpi,omp,creationTime,primTime,comunicationTime,totalTime=line.strip()[:-1].split(";")
        if type not in data:
            data[type]={}
            num[type]={}
        if mpi not in data[type]:
            data[type][mpi]={}
            num[type][mpi]={}
        if omp not in data[type][mpi]:
            data[type][mpi][omp]=[float(creationTime),float(primTime),float(comunicationTime), float(totalTime)]
            num[type][mpi][omp]=1
        else:
            data[type][mpi][omp][0]+=float(creationTime)
            data[type][mpi][omp][1]+=float(primTime)
            data[type][mpi][omp][2]+=float(comunicationTime)
            data[type][mpi][omp][3]+=float(totalTime)
            num[type][mpi][omp]+=1
    f.close()#close the source file and open the result file
    f=open(resultFile,"w")
    f.write("Modality;MPI;OMP;Graph Creation Time;Prim Execution Time;Comunication Time;Total Time;\n")
    for typeKey in data.keys():#for each element in the dictionary calculate the mean and save it in the result file
        ompList=list(data[typeKey].keys())
        ompList.sort()
        for mpiKey in ompList:
            ompList=list(data[typeKey][mpiKey].keys())
            ompList.sort()
            for ompKey in ompList:
                it=num[typeKey][mpiKey][ompKey]
                data[typeKey][mpiKey][ompKey][0]/=it
                data[typeKey][mpiKey][ompKey][1]/=it
                data[typeKey][mpiKey][ompKey][2]/=it
                data[typeKey][mpiKey][ompKey][3]/=it
                f.write(typeKey+";"+str(mpiKey)+";"+str(ompKey)+";"+str(data[typeKey][mpiKey][ompKey][0])+
                ";"+str(data[typeKey][mpiKey][ompKey][1])+";"+str(data[typeKey][mpiKey][ompKey][2])+
                ";"+str(data[typeKey][mpiKey][ompKey][3])+";\n")
    f.close()#close the result file

if os.path.exists("TimeResults"):#if Result directory exist, delete it and all its elements
    shutil.rmtree('TimeResults')

os.mkdir("TimeResults")#create the result directory

for path, currentDirectory, files in os.walk("Information"):#for each file in the directory Informations, analize them and save the results in the relative path in Results directory 
    for file in files:
        resultPath=path.replace("Information","TimeResults")
        if not(os.path.exists(resultPath)):
            os.makedirs(resultPath)
        analizeFile(os.path.join(path, file),os.path.join(resultPath,file))