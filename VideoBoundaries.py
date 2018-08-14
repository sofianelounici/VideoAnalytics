from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import peakutils
from scipy import signal    
from os.path import isfile, join


class VideoBoundaries:
    """ Class VideoBoundaries : Apply methods for shot segmentation using ECR"""
    def __init__(self):
        self.fps=0
        self.listECR=0
        self.listChangePoints=0
    
    def frame(self, number_frames, video):
        """ Convert video into gray frames"""
        cap = cv2.VideoCapture(video)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
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
          
    def convert_frames_to_video(self, pathIn,pathOut):
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
 
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), self.fps, size)
 
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
    
    def checkMotion(self,y,pi,threshold, step):
        """ Returns a boolean value to decide if the peak is due to a motion"""
        isInMotion=False
        t=[y[pi+j] for j in range(-step,0)]
        closePeaks=0
        # We observe the a defined number of frames before the peak
        for h in t:
            if h>y[pi]*(0.75): # If we detect peak with comparable level of intensity
                closePeaks+=1
        if closePeaks>=len(t)/2: # If a certain amount of peaks with comparable level of intensity
            isInMotion=True
        return isInMotion
    
    def peakId(self, y, threshold, step):
        """ Returns the list of peaks in the ECR serie"""
        p = peakutils.indexes(np.array(y), thres=threshold, min_dist=10)
        listP=[p[0]]
        for i in range(1, len(p)):
            # We check that the peak is not due to a motion in the image
            if(self.checkMotion(y, p[i], threshold, step)==False):
                listP.append(p[i])
        return listP
    
    def pooling(self, t, nb):
        """ Returns a neighbor-average of the ECR series"""
        for i in range(nb):
            newT=[]
            for i in range(1,len(t)-1):
                newT.append(max(t[i-1], t[i], t[i+1]))
            t=newT.copy()
        return newT
    
    def detectCut(self, thres, step):
        """ Returns the list of changepoints based on threshold method"""
        divd = lambda x,y: 0 if y==0 else x/y
        n,m = self.listImage[0].shape
        self.listECR=[]
        # Ratio ECR(n-1,n) / ECR(n-10,n)
        for i in range(1, len(self.listImage)):
            t=self.ECR(self.listImage[i-1],self.listImage[i], n, m, 5)
            if(i>10):
                tDelayed = self.ECR(self.listImage[i-10],self.listImage[i], n, m, 5)
                self.listECR.append(t*(1+tDelayed))
            else:
                self.listECR.append(t)
        ecr = self.pooling(self.displayECR(), 2) #Pooling Operation
        self.listChangePoints = self.peakId(ecr, thres, step) #Peak Detection
        
    
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
                start= self.listChangePoints[j-1]
                end= self.listChangePoints[j]
            if(j==0):
                start=0
                end= self.listChangePoints[0]
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
            self.convert_frames_to_video(toConvert,out)
            command = "rm  -rf scene"+str(j)
            os.system(command)
        if(verbose):
            print("Done !")
            
    def videoCut(self):
        """ Returns the cutted video"""
        return self.VideoCut
    def accuracy(self, results, tolerance, verbose):
        actualCP=[]
        lines = [int(line.rstrip('\n'))-1 for line in open(results)][::-1]
        false=0
        missed=0
        correct=0
        # Check if the value in a tolerance range is detected
        for h in self.listChangePoints:
            t=[h+i for i in range(-tolerance,tolerance+1)]
            isIn=False
            v=0
            for f in t:
                if f in lines: # Correct case
                    isIn=True 
                    v=f
                    break
            if(isIn==False): # False position case
                false=false+1
            else:
                lines.remove(v)
                correct=correct+1
                
        missed=len(lines) # Number of shots non  detected

        recall = correct/(correct+missed)
        precision = correct/(correct+false)
        f1 = 2*precision*recall/(precision+recall)
        
        if(verbose):
            print("With ",tole,"frame(s) of tolerance :\n")
            print("Recall : ",recall)
            print("Precision :",precision)
            print("F1 :",f1)
        return recall, precision, f1