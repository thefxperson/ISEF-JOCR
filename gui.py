from aqt import mw
from aqt.utils import showInfo
from aqt.qt import *

class closableDialogue(QDialog):
	def reject(self):
		QDialog.reject(self)


def testopen(mw):
	dia = closableDialogue()	#create new window with properties from closableDialogue class
	dia.finished.connect(lambda: dia)	#keep the window open through black magic and my thoughts and prayers
	dia.show()					#show the window

#new item called test
action = QAction("test",mw)
#when activated run testopen
action.triggered.connect(testopen)
#add test to the Tools menu in Anki
mw.form.menuTools.addAction(action)