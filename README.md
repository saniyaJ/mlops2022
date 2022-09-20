Pushing git, with requirement.txt having packages required.


Python version screenshot is. PythonV.png

Output of classification is : Output.png

NLTK Version :nltkV.png

Hyper Parameter tunning 
1) Images are tuned with various parameters of gamma, C. Best accuracy comes at gamma =0.001, C=5 as 0.95

Changing image resolution
1) rescale is applied at first, gives best accuracy at gamma =0.005, C=15 as 0.88
2) downscale_local_mean is applied , gives best accuracy at gamma =0.005, C=0.6 as 0.86
3) resize is applied , gives best accuracy at gamma =0.005, C=10 as 0.74

Scnreenshots gives detail of accuracy at all level