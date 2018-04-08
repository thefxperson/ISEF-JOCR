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

def getContext(term, weight, freq):
	sumvec = np.zeros(30)
	kana = np.load("chars/kana_list.npy").item()
	kandict = np.load("chars/kanji_list.npy").item()

	if term in kana:
		sumvec += (.25 * kandict[term])		#.5 (if you search "a", you would get hiragana a, and katakana a, so 1/2) --tuned down to .25 since it was too much
	else:
		#search the item on jisho and process the return
		foo = jisho.jisho()
		foo.search(term)
		s = ""
		common = {}
		for i in range(foo.num_entries):
			if i <= 5:
				word = foo.getWord(i)
				for j in range(foo.num_jp_entries):
					if word[j] != None:
						s += word[j]
						common[word[j]] = foo.is_common[i]

		#get frequency for every character
		classcount = {}
		for i in s:
			if i not in classcount:
				classcount[i] = 1
			else:
				classcount[i] += 1

		sort = sorted(classcount, key=classcount.get)

		#compute vector of sum of percents
		pct = classcount[sort[-freq]]/len(s)
		if sort[-freq] in kandict:
			if common[sort[-freq]]:						#common words are worth 2.5 times as much as uncommon words
				sumvec += 2.5 * (pct * kandict[sort[-freq]])
			else:
				sumvec += (pct * kandict[sort[-freq]])

	return sumvec * weight

def combineOut(output, context):	#combine NN output with context output, return label
	out = output + context
	out = np.split(out, 6)
	out = softmax(out)
	foo = []
	for i in range(len(out)):
		foo.append(np.argmax(out[i]))
	return fiveToAlpha(foo)

#kanAlpha = np.load("chars/kanji_list.npy").item()
#alphaKan = np.load("chars/output_list.npy").item()

#print(alphaKan[combineOut(getContext("day", 1, 1), getContext("day", 1, 1))])
#print(alphaKan["亜"])