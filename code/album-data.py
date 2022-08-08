import pandas as pd
import requests

df = pd.read_csv("../MSD-I_dataset.tsv", sep="\t")

#split data into training and testing
# train, test = train_test_split(df, test_size=0.2)
#should i loop in the validation set with testing?
train = df[df['set'] == 'train']
test = df[df['set'] == 'test']
val = df[df['set'] == 'val']
# genres = df['genre'].unique()

# converts column of album urls to a Python list
train_albums = train['image_url'].tolist()
test_albums = test['image_url'].tolist()
val_albums = val['image_url'].tolist()
#extract list of genres for train and test data
train_genres = train['genre'].tolist()
test_genres = test['genre'].tolist()
val_genres = val['genre'].tolist()

print("number of training images: ", len(train_albums))
print("number of testing images: ", len(test_albums))
print("number of validation images: ", len(val_albums))

# using requests to save images from each image url 
# reference to solution: https://stackoverflow.com/questions/30229231/python-save-image-from-url

for i in range(len(train_albums)):
    url = train_albums[i]
    genre = train_genres[i]
    image_name = "image_" + str(i) + ".jpg"
    dir = "../data/train/" + genre + "/" + image_name
    with open(dir, 'wb') as handler:
        response = requests.get(url).content
        handler.write(response)

for j in range(len(test_albums)):
    url = test_albums[j]
    genre = test_genres[j]
    image_name = "image_" + str(j) + ".jpg"
    dir = "../data/test/" + genre + "/" + image_name
    with open(dir, 'wb') as handler:
        response = requests.get(url).content
        handler.write(response)

for k in range(len(val_albums)):
    url = val_albums[k]
    genre = val_genres[k]
    image_name = "image_" + str(k) + ".jpg"
    dir = "../data/val/" + genre + "/" + image_name
    with open(dir, 'wb') as handler:
        response = requests.get(url).content
        handler.write(response)