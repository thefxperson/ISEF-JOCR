import numpy as np
import jisho

def fiveToAlpha(label):            #predictions[30], labels[6]
		#convert from alpha to one hot
		one = []
		foo = label
		for i in range(len(label)):         #converts to labels[6][5]
			if foo[i] == 0:
				one.append("a")
			elif foo[i] == 1:
				one.append("b")
			elif foo[i] == 2:
				one.append("c")
			elif foo[i] == 3:
				one.append("d")
			elif foo[i] == 4:
				one.append("e")
		return "".join(one)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getContext(term, weight):
	#search the item on jisho and process the return
	foo = jisho.jisho()
	foo.search(term)
	s = ""
	for i in range(foo.num_entries):
		word = foo.getWord(i)
		for j in range(foo.num_jp_entries):
			if word[j] != None:
				s += word[j]

	#get frequency for every character
	classcount = {}
	for i in s:
		if i not in classcount:
			classcount[i] = 1
		else:
			classcount[i] += 1

	#compute vector of sum of percents
	sumvec = np.zeros(30)
	kandict = np.load("kanji_list.npy").item()
	for key in classcount:
		pct = classcount[key]/len(s)
		if key in kandict:
			sumvec += (pct * kandict[key])
	sumvec = np.split(sumvec, 6)
	sumvec = softmax(sumvec)
	sumvec = sumvec * weight
	return np.reshape(sumvec, 30)

def combineOut(output, context):	#combine NN output with context output, return label
	out = output + context
	out = np.split(out, 6)
	out = softmax(out)
	foo = []
	for i in range(len(out)):
		foo.append(np.argmax(out[i]))
	return fiveToAlpha(foo)