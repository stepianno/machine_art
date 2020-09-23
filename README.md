# machine_art
Generating Images through a CNN  

In this project I create and train an image Autoencoder - using over 10,000 images scraped from Pinterest - in order to compress an image into an encoded file 
a tenth the size of the original before then decoding it to recreate the original image. Once trained, I use the Autoencoder for two separate purposes:  

The first objective is to create brand new, unique images. I scrape over 2,000 paintings from the Museum of Modern Art (MoMA) and encode each of them. Once encoded,
I cluster the encodings and aggregate each cluster by taking the maximum, mean, or minimum values of each feature in the cluster. I then put that aggregation through
the decoder half of the Autoencoder to produce a brand new image. Through this process I create several unique, fascinating works to behold, all available in this 
repository. The Flask App includes a virtual gallery which allows the user to peruse through the best of these!  

The second objective is a recommendation app which allows a user to input an image and get the top three recommended art pieces based on the input image. This is 
accomplished by encoding the inputed image and calculating the cosine distance of the image to every encoded image in my modern art corpus. The three closest 
paintings are then displayed to contrast with the input image. This app is also part of the Flask App included in this repository.  

If interested in making use of the app, simply fork this repository, cd into flask_app and run the command: python art.py  

I hope you enjoy exploring this work as much as I've enjoyed making it!
