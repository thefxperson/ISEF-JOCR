#creates and manages the gui. gives input values to ui.py to conjugate

import kivy
kivy.require('1.10.0')

import jisho
import random

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle

class GUI(GridLayout):
	firsttime = True
	recentSearch = []
	entry = 0


	def __init__(self, **kwargs):
		super(GUI, self).__init__(**kwargs)
		self.menu(self)
		
	def menu(self, obj):
		self.clear_widgets()
		self.cols = 1

		self.searchText = TextInput(hint_text="Enter Search Here", multiline=False)
		self.add_widget(self.searchText)

		self.searchBtn = Button(text="Search")
		self.searchBtn.bind(on_release=self.search)
		self.add_widget(self.searchBtn)

		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))
		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))
		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))
		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))
		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))
		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))
		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))
		self.add_widget(Label(text='[color=FFFFFF]null[/color]', markup=True))


	def search(self, obj):
		self.clear_widgets()
		self.cols = 1

		self.backBtn = Button(text="Back")
		self.backBtn.bind(on_release=self.menu)
		self.add_widget(self.backBtn)

		self.foo = jisho.jisho()
		self.foo.search(self.searchText.text)

		self.results = []

		for i in range(0,self.foo.num_entries):
			self.word = self.foo.getWord(i)
			self.read = self.foo.getReading(i)
			if self.word[0] != "NONE":
				self.results.append(Button(font_name="TakaoPMincho.ttf",text="[color=FFFFFF][size=36]" + self.word[0] + "[/size][/color]",markup=True))
			elif self.read[0] != "NONE":
				self.results.append(Button(font_name="TakaoPMincho.ttf",text="[color=FFFFFF][size=36]" + self.read[0] + "[/size][/color]",markup=True))
			else:
				self.results.append(Button(font_name="TakaoPMincho.ttf",text="[color=FFFFFF][size=36]No Results Found.[/size][/color]",markup=True))
			self.results[i].bind(on_release=self.info)
			self.add_widget(self.results[i])

	def info(self, obj):
		for i in range(0, len(self.results)):
			if self.results[i] == obj:
				self.entry = i

		self.clear_widgets()
		self.cols = 1

		self.backBtn = Button(text="Back")
		self.backBtn.bind(on_release=self.search)
		self.add_widget(self.backBtn)

		word = "[color=FFFFFF][size=36]"
		read = self.foo.getReading(self.entry)

		for i in range(0, len(read)):
			if i != (len(read)-1) and read[i] != "NONE":
				word = word + read[i] + " - "
			elif read[i] != "NONE": word += read[i]
		word += "[/size][/color]"

		self.add_widget(Label(halign="center",text_size=(self.width, None), font_name="TakaoPMincho.ttf",text=word, markup=True))

		word = "[color=FFFFFF][size=36]"
		words = self.foo.getWord(self.entry)

		for i in range(0, len(words)):
			if i != (len(words) - 1) and words[i] != "NONE":
				word = word + words[i] + " - "
			elif words[i] != "NONE": word += words[i]
		word += "[/size][/color]"

		self.add_widget(Label(halign="center",text_size=(self.width, None), font_name="TakaoPMincho.ttf",text=word, markup=True))

		word = "[color=FFFFFF][size=36]"
		definitions = self.foo.getDef(self.entry)

		for i in range(0, len(definitions)):
			if i != (len(definitions) - 1) and definitions[i] != "NONE":
				word = word + definitions[i] + ", "
			elif definitions[i] != "NONE": word += definitions[i]
		word += "[/size][/color]"

		self.add_widget(Label(halign="center",text_size=(self.width, None), font_name="TakaoPMincho.ttf",text=word, markup=True))

		self.cardBtn = Button(font_name="TakaoPMincho.ttf",text="[color=FFFFFF][size=36]Make Flashcard[/size][/color]",markup=True)
		self.cardBtn.bind(on_release=self.makeCard)
		self.add_widget(self.cardBtn)

	def makeCard(self,obj):
		file = open("flashcards.txt","r+")
		if file.readline() == "tags:j3":
			print(test)
		file.close()


class JishoApp(App):
	
	def build(self):

		return GUI()

if __name__ == "__main__":
	JishoApp().run()