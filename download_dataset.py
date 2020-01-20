import wget
import os
import shutil
import gzip

#DIRECTORIES AND FILES SETUP
dataset_raw_location = "./dataset/"
images_file_name = "train-images-idx3-ubyte"
labels_file_name = "train-labels-idx1-ubyte"

#Clean eventual leftovers in directory
if os.path.isdir(dataset_raw_location) == False:
    os.mkdir(dataset_raw_location)
else:
    if os.path.isfile(dataset_raw_location + images_file_name): os.remove(dataset_raw_location + images_file_name)
    if os.path.isfile(dataset_raw_location + labels_file_name): os.remove(dataset_raw_location + labels_file_name)
    if os.path.isfile(dataset_raw_location + images_file_name + ".gz"): os.remove(dataset_raw_location + images_file_name + ".gz")
    if os.path.isfile(dataset_raw_location + labels_file_name + ".gz"): os.remove(dataset_raw_location + labels_file_name + ".gz")

#Download dataset archives
wget.download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", dataset_raw_location)
wget.download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", dataset_raw_location)

if os.path.isfile(dataset_raw_location + images_file_name + ".gz") and os.path.isfile(dataset_raw_location + labels_file_name + ".gz"):
    #Extract images binary dataset from archive
    input = gzip.GzipFile(dataset_raw_location + images_file_name + ".gz", 'rb')
    images = input.read()
    input.close()
    output = open(dataset_raw_location + images_file_name, 'wb')
    output.write(images)
    output.close()
    #Extract labels binary dataset from archive
    input = gzip.GzipFile(dataset_raw_location + labels_file_name + ".gz", 'rb')
    labels = input.read()
    input.close()
    output = open(dataset_raw_location + labels_file_name, 'wb')
    output.write(labels)
    output.close()
    #Remove *.gz files
    os.remove(dataset_raw_location + images_file_name + ".gz")
    os.remove(dataset_raw_location + labels_file_name + ".gz")
    print("\n Downloaded and extracted dataset!")
else:
    print("\n Couldn't download or extract dataset.")
