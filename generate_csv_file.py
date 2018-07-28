import numpy as np
import csv

inc = np.load("inception_predictions.npy")
xce = np.load("xception_predictions.npy")
inr = np.load("inceptionresnet_predictions.npy")
res = np.load("resnet_predictions.npy")
den = np.load("densenet_predictions.npy")
print(den.shape)
fnames = np.load("filenames.npy")

print(fnames)

f = open("test_predictions.csv", "w")
wtr = csv.writer(f)
wtr.writerow(["image","MEL","NV","BCC","AKIEC","BKL","DF","VASC"])
for i in range(fnames.shape[0]):
    wtr.writerow([fnames[i]] + ((
        inc[i] + xce[i] + inr[i] + res[i] + den[i]
    )/5).tolist())
f.close()
