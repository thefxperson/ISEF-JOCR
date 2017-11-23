import os

#Uncomment following area to extract images from training set
'''
alphabets = os.listdir("images_background/")
alphabets = alphabets[1:]

for i in range(len(alphabets)):
	chars = os.listdir("images_background/" + alphabets[i] + "/")
	chars = chars[1:]
	for j in range(len(chars)):
		imgs = os.listdir("images_background/" + alphabets[i] + "/" + chars[j] + "/")
		for k in range(len(imgs)):
			os.rename(("images_background/" + alphabets[i] + "/" + chars[j] + "/" + imgs[k]), imgs[k])
	print(i+1, "alphabets done out of", len(alphabets),"alphabets.")
'''

#Uncomment following area to extract images from testing set
'''
alphabets = os.listdir("images_evaluation/")
alphabets = alphabets[1:]

for i in range(len(alphabets)):
	chars = os.listdir("images_evaluation/" + alphabets[i] + "/")
	chars = chars[1:]
	for j in range(len(chars)):
		imgs = os.listdir("images_evaluation/" + alphabets[i] + "/" + chars[j] + "/")
		for k in range(len(imgs)):
			os.rename(("images_evaluation/" + alphabets[i] + "/" + chars[j] + "/" + imgs[k]), imgs[k])
	print(i+1, "alphabets done out of", len(alphabets),"alphabets.")
'''