import os
from unittest.mock import DEFAULT
import cv2
def main(c):
    if(c==1):
        print("enter image path: ")
        img=cv2.imread(input(" ")) 
        p2(img,0.5)
    if(c==2):
        print("enter image path: ")
        img=cv2.imread(input(" ")) 
        p2(img,0.3)
    else:
        os.system("exit")


def banner():
    print("""\33[96m
██╗ ██████╗ ███╗   ██╗
██║██╔═══██╗████╗  ██║
██║██║   ██║██╔██╗ ██║
██║██║   ██║██║╚██╗██║
██║╚██████╔╝██║ ╚████║
╚═╝ ╚═════╝ ╚═╝  ╚═══╝\33[92m
github:amine10a
fb:emine.ardhaoui """)
def choix():
    banner()
    print("""
    [1]Selct image objects
    [2]Select deep image objects
    [3]exit""")
    while(True):
        c=int(input("enter your choix: "))
        if(0<c<3):return c


def p2(img,sel):
    h, w, c = img.shape
    print(h," ", w," ",c)
    if(h>1200 and w>1200):
        res = cv2.resize(img, (w//10,h//10))
        img = res



    classnames = [] 
    classfile  = 'files/thing.names'

    with open(classfile, 'rt') as f :
      classnames = f.read().rstrip('\n').split('\n')
    p = 'files/frozen_inference_graph.pb'
    v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

    net = cv2.dnn_DetectionModel(p,v) 
    net.setInputSize(320,240)      

    net.setInputScale(1.0/127.5)     
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)         

    classIds ,confs , bbox = net.detect(img, confThreshold=sel)
    print(classIds,bbox)
    for classId , confidence , box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=3)
        cv2.putText(img,classnames[classId-1],
                    (box[0]+10,box[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2)
    

    cv2.imshow('ION', img)
    cv2.waitKey(0)
    msg = cv2.imwrite('/images/ion_1.jpg', img)
    if(msg==False):
        print("image is not saved !")
    if(msg==True):
        print("image is  saved ^")
c=choix()
main(c)
