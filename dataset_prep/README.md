# Preparing the ISIC Dataset (Task 3)

An outline of the steps I took to prepare the ISIC data for training a model for
task 3.

### 1. Downloaded HAM10000
First I downloaded the following files:
  - ISIC2018_Task3_training_GroundTruth
  - ISIC2018_Task3_training_Input

These constituted the HAM10000 dataset explicitly mentioned by the challenge.

### 2. Sorted the HAM10000 by Class
Next, I ran the `util_sort_isic_input.py` script to sort the HAM files by which
class they belong to.

### 3. Split the HAM10000 into training and validation sets
I split the HAM10000 into 90% training, 10% validation in order to use as much
of the data as I could for training

### 4. Downloaded the rest of the ISIC Archive
Next, I downloaded every file from ISIC that wasn't included in the HAM10000
along with its metadata.

### 5. Sorted the rest of the ISIC Archive by Class
Next, I rand the `use_unlabeled_meta.py` script to sort the rest of the data
into classes when that information was given in the accompanying JSON, and when
the diagnosis matched one of the 7 for this challenge. I put the remainder in an
"unlabeled folder" for the purposes of semi-supervised learning if time allows.
