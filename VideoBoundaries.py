from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import peakutils
from scipy import signal    
from os.path import isfile, join


class VideoBoundaries:
    
    def __init__(self):
		""" Performs video boundary detection, using ECR (Edge Change Ratio)"""
        pass
    
    def frame(self, number_frames, video):
        """ Convert video into gray frames"""
        cap = cv2.VideoCapture(video)
        i=0
        listImage=[]
        listHist=[]
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while i<min(number_frames, length):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            listImage.append(gray)
            i=i+1
        self.listImage=listImage
        cap.release()
        cv2.destroyAllWindows()
        
    @staticmethod    
    def convert_frames_to_video(pathIn,pathOut,fps):
        """Convert frames into a video"""
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        #for sorting the file names properly
        files.sort(key = lambda x: int(x[5:-4]))
 
        for i in range(len(files)):
            filename=pathIn + '\/' + files[i]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            #inserting the frames into an image array
            frame_array.append(img)
 
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()
        
    @staticmethod
    def ECR(prevFrame, currFrame, width, height, dilate_rate):
        """ Apply the Edge Change Ratio Algorithm"""
        divd = lambda x,y: 0 if y==0 else x/y
    
        edgePrev=cv2.Canny(prevFrame, 0, 200)
        inv_dilatedPrev=(255-cv2.dilate(edgePrev, np.ones((dilate_rate, dilate_rate))))
    
        edgeCurr=cv2.Canny(currFrame, 0, 200)
        inv_dilatedCurr=(255-cv2.dilate(edgeCurr, np.ones((dilate_rate, dilate_rate))))
    
        lPrev = (edgePrev & inv_dilatedCurr)
        lCurr = (edgeCurr & inv_dilatedPrev)
    
        sumPrev = np.sum(edgePrev)
        sumCurr= np.sum(edgeCurr)
    
        outPixels = np.sum(lPrev)
        inPixels = np.sum(lCurr)
    
        return max(divd(float(inPixels), float(sumCurr)), divd(float(outPixels), float(sumPrev)))
    
    
    def displayECR(self):
        return self.listECR
    
    def peakId(self, y, threshold, minFrame):
        """ Returns the list of peaks in the ECR serie"""
        p = peakutils.indexes(y, thres=threshold, min_dist=1)
        listP=[p[0]]
        for i in range(1, len(p)):
            if(p[i]-listP[-1]>minFrame):
                listP.append(p[i])
        return listP
    
    def detectCut(self, minFrame):
        """ Returns the list of changepoints based on threshold method"""
        divd = lambda x,y: 0 if y==0 else x/y
        n,m = self.listImage[0].shape
        self.listECR=[]
        for i in range(1, len(self.listImage)):
            self.listECR.append(self.ECR(self.listImage[i-1],self.listImage[i], n, m, 5))
        ecr = self.displayECR()
        self.listChangePoints = self.peakId(ecr, 0.6, minFrame)   
        
    
    def extractClip(self, where, verbose=False):
        """ Extracts hard cuts from a video"""
        cwd = os.getcwd()
        if(verbose):
            print("End of Computing Cuts\n")
            print(len(self.listChangePoints)," cuts detected\n")
        command = "mkdir "+where
        os.system(command)
        self.VideoCut=[]
        for j in range(len(self.listChangePoints)):
            command = "mkdir scene"+str(j)
            os.system(command)
            start=0
            end=0
            if(j<len(self.listChangePoints)-1 and j>0):
                start= self.listChangePoints[j]
                end= self.listChangePoints[j+1]
            if(j==0):
                start=0
                end= self.listChangePoints[j+1]
            if(j==len( self.listChangePoints)-1):
                start= self.listChangePoints[j]
                end=len(self.listImage)
            
            self.Img=[]
            for i in range(start, end):
                path=cwd+"\scene"+ str(j)
                imname = "image"+str(i)+".png"
                cv2.imwrite(os.path.join(path, imname), self.listImage[i])
                self.Img.append(self.listImage[i])
                
            self.VideoCut.append(self.Img)
                
            out=where +'\\video'+ str(j)+'.mp4'
            toConvert = cwd+"\scene"+ str(j)
            self.convert_frames_to_video(toConvert,out,24)
            command = "rm  -rf scene"+str(j)
            os.system(command)
        if(verbose):
            print("Done !")
            
    def videoCut(self):
        """ Returns the cutted video"""
        return self.VideoCut