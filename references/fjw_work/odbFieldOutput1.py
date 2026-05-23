#! /user/bin/python
#- -coding: UTF-8-*-
#odbFiledOutput.py
# čŻťĺčžĺşć°ćŽĺş? *.odbçĺĺ˛ć°ć?
#ĺŻźĺĽabaqus odbAcessć¨Ąĺ
from odbAccess import *
from abaqusConstants import *
import time
#čˇĺmatlabćĺŽçć°ćŽĺşďźé¨äťśĺďźčçšďźçšĺŽçťćçäżĄć?
path='vert.odb'
ReqData='U'
step='Load'
DataFile=open(ReqData+'1.txt','w')
#ćĺźodbć°ćŽĺş?
myodb = openOdb(path=path)
val = myodb.steps[step].frames[-1].fieldOutputs[ReqData].values
#for i in range(0,45020):
#for i in range(0,636421):
for i in range(0,len(val)):
    seq=val[i].nodeLabel
    u1=val[i].data[0]
    u2=val[i].data[1]
    u3=val[i].data[2]
    DataFile.write('%10.6E\t'%seq);
    DataFile.write('%10.6E\t'%u1);
    DataFile.write('%10.6E\t'%u2);
    DataFile.write('%10.6E\n'%u3);
DataFile.close()
myodb.close()