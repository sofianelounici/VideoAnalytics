""" Boundary detection on the GoldenEye video"""

goldeneye = VideoBoundaries()
goldeneye.frame(2000,'goldeneye.mp4')
goldeneye.detectCut(15)

#Your output path
goldeneye.extractClip(r'C:\Users\Sofiane\Desktop\Projects-Notebooks\SceneDetection\video', verbose=True)