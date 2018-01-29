import requests
import numpy as np

class jisho:
	response = {}
	japanese_word = {}
	japanese_reading = {}
	english_definition = {}
	is_common = {}

	num_entries = 0
	num_en_entries = 0
	num_jp_entries = 0

	def search(self, keyword):
		self.clearVals()
		self.response = requests.get("http://jisho.org/api/v1/search/words?keyword=" + keyword)
		if self.response.status_code != 200: print("API Error: {}".format(self.response.status_code))

		for a, i  in enumerate(self.response.json()["data"]):
			self.is_common[a] = i["is_common"]
			for b, j in enumerate(i["japanese"]):
				if "word" in j:
					self.japanese_word[a,b] = j["word"]
				if "reading" in j:
					self.japanese_reading[a,b] = j["reading"]
				if self.num_jp_entries < b+1:
					self.num_jp_entries = b+1

			for c, k in enumerate(i["senses"][0]["english_definitions"]):
				self.english_definition[a,c] = k
				
				if self.num_en_entries < c+1:
					self.num_en_entries = c+1

			if self.num_entries < a+1:
				self.num_entries = a+1

	def disp(self):
		num = self.num_entries
		jpnum = self.num_jp_entries
		ennum = self.num_en_entries
		for i in range(0,num):
			for j in range(0,jpnum):
				if self.japanese_word.get((i,j),"") != "":
					print(self.japanese_word.get((i,j),"") + " ",end="")
			
			if self.japanese_word.get((i,0),"") != "":
				print("- " + self.japanese_reading.get((i,0),""))
			else:
				print(self.japanese_reading.get((i,0),""))
			for k in range(0,ennum):
				if self.english_definition.get((i,k),"") != "":
					if self.english_definition.get((i,k+1),"") != "":
						print(self.english_definition.get((i,k),"") + ", ", end="")
					else:
						print(self.english_definition.get((i,k),"") + "\n")

	def getWord(self, entry):
		words = []
		for i in range(0,self.num_jp_entries):
			if self.japanese_word.get((entry,i), None) != None:
				words.append(self.japanese_word[entry,i])
			else:
				words.append(None)
		return words

	def getReading(self, entry):
		words = []
		for i in range(0,self.num_jp_entries):
			if self.japanese_reading.get((entry,i), None) != None:
				words.append(self.japanese_reading[entry,i])
			else:
				words.append(None)
		return words

	def getDef(self, entry):
		words = []
		for i in range(0,self.num_en_entries):
			if self.english_definition.get((entry,i), None) != None:
				words.append(self.english_definition[entry,i])
			else:
				words.append(None)
		return words

	def clearVals(self):
		self.response.clear()
		self.japanese_word.clear()
		self.japanese_reading.clear()
		self.english_definition.clear()
		self.is_common.clear()



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getContext(term, weight):
	#search the item on jisho and process the return
	foo = jisho()
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

temp = getContext("eye", 2)
print(temp)