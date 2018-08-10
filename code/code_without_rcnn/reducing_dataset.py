import os
import shutil

image_path1 = 'raw_data/flickr30k-images/'
#image_path2 = 'raw_data/Flicker8k_Dataset/'

images1 = os.listdir(image_path1)
#images2 = os.listdir(image_path2)

cap_path1 = 'raw_data/results_20130124.token'
#cap_path2 = 'raw_data/Flickr8k.lemma.token.txt'

with open(cap_path1, 'r') as f1:
    data1 = f1.readlines()

#with open(cap_path2, 'r') as f2:
#    data2 = f2.readlines()

captions1 = [caps.replace('\n', '').split('\t')[1] for caps in data1]
#captions2 = [caps.replace('\n', '').split('\t')[1] for caps in data2]


new_captions1 = []
#new_captions2 = []
for i in range(0,len(captions1)):
    if captions1[i].find(' dog')>0:
        new_captions1.append(data1[i])

#for i in range(0,len(captions2)):
#    if captions2[i].find(' dog')>0:
#        new_captions2.append(data2[i])

#filenames1 = [image_path1+caps.split('\t')[0].split('#')[0] for caps in new_captions1]
#filenames2 = [image_path2+caps.split('\t')[0].split('#')[0] for caps in new_captions2]

filenames1 = [image_path1+caps.split('\t')[0] for caps in new_captions1]
#filenames2 = [image_path2+caps.split('\t')[0] for caps in new_captions2]

captions1 = [caps.replace('\n', '').split('\t')[1] for caps in new_captions1]
#captions2 = [caps.replace('\n', '').split('\t')[1] for caps in new_captions2]

captions = []
for i in range(0,len(new_captions1)):
    captions.append(filenames1[i]+'\t'+captions1[i])

#for i in range(0,len(new_captions2)):
#    captions.append(filenames2[i]+'\t'+captions2[i])


#print(len(captions))

training_n = int(0.8*len(captions))-2
val_n = int(0.1*len(captions))-2

training_captions = captions[0:training_n]
#print(training_captions[training_n-1])

validation_captions = captions[training_n:training_n+val_n]
#print(validation_captions[0])
#print(validation_captions[val_n-1])

test_captions = captions[training_n+val_n:]
#print(test_captions[0])
#print(test_captions[val_n-1])


#print(len(training_captions))
#print(len(validation_captions))
#print(len(test_captions))



filenames = [caps.split('\t')[0].split('#')[0] for caps in training_captions]
filenames = set(filenames)

for i in range(0,len(training_captions)):
    training_captions[i] = 'Training/'+training_captions[i].split('/')[2]

src = os.path.dirname(os.path.abspath(__file__))

for file_name in filenames:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, 'Data/Training')

write_file = 'Data/training.txt'
fi = open(write_file, 'w')

for item in training_captions:
  fi.write("%s\n" % item)




filenames = [caps.split('\t')[0].split('#')[0] for caps in validation_captions]
filenames = set(filenames)
for i in range(0,len(validation_captions)):
    validation_captions[i] = 'Validation/'+validation_captions[i].split('/')[2]

src = os.path.dirname(os.path.abspath(__file__))

for file_name in filenames:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, 'Data/Validation')

write_file = 'Data/validation.txt'
fi = open(write_file, 'w')

for item in validation_captions:
  fi.write("%s\n" % item)




filenames = [caps.split('\t')[0].split('#')[0] for caps in test_captions]
filenames = set(filenames)
for i in range(0,len(test_captions)):
    test_captions[i] = 'Test/'+test_captions[i].split('/')[2]

src = os.path.dirname(os.path.abspath(__file__))

for file_name in filenames:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, 'Data/Test')

write_file = 'Data/test.txt'
fi = open(write_file, 'w')

for item in test_captions:
  fi.write("%s\n" % item)




"""
print(len(filenames1))
print(len(filenames2))

print(len(new_captions1))
print(len(new_captions2))
print(len(fin_captions))
"""

#Splitting datasets into training, validation and test




