""" Boundary detection on the GoldenEye video"""

videoToDetect = VideoBoundaries()
videoToDetect.frame(2000,'goldeneye.mp4')
videoToDetect.detectCut(thres=0.6, minFrame=15)

#Your output path
goldeneye.extractClip(r'C:\Users\Sofiane\Desktop\Projects-Notebooks\SceneDetection\video', verbose=True)
# Compute accuracy with a tolerance level concerning the frames
recall, precision, f1 = goldeneye.accuracy(r'C:\Users\Sofiane\Desktop\Projects-Notebooks\VideoAnalytics\test_results_1600fr.txt', tolerance=2, verbose=True)