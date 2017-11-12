from aqt import mw
from aqt.utils import showInfo
from aqt.qt import *

class jocrWindow(QWidget):
	def __init__(self):
		super(jocrWindow, self).__init__()

		self.setGeometry(300,300,400,400)	#x,y on screen, width and height
		self.setWindowTitle("jocr")

		self.layout = QGridLayout()


		self.top3 = QLabel("top 3: A B C")
		self.layout.addWidget(self.top3,0,1,1,4)

		#-------------------------------------

		self.left = QPushButton("<-",self)
		self.left.setMinimumHeight(300)
		self.layout.addWidget(self.left,1,0)
		
		self.canvas = QLabel("canvas")
		self.layout.addWidget(self.canvas,1,1,1,4)

		self.right = QPushButton("->",self)
		self.right.setMinimumHeight(300)
		self.layout.addWidget(self.right,1,5)

		#--------------------------------------

		self.override = QLabel("Override:")
		self.layout.addWidget(self.override,2,1)

		self.red = QPushButton("<10m",self)
		self.layout.addWidget(self.red,2,2)

		self.green = QPushButton("10m",self)
		self.layout.addWidget(self.green,2,3)

		self.blue = QPushButton("3d",self)
		self.layout.addWidget(self.blue,2,4)


		self.layout.setRowStretch(1,1)

		self.setLayout(self.layout)


	#def reject(self):
	#	QDialog.reject(self)


#def testopen(mw):
#	self.dia = jocrWindow()	#create new window with properties from jocrWindow class
	#dia.finished.connect(lambda: dia)	#keep the window open through black magic and my thoughts and prayers
	#dia.show()					#show the window

def testShow():
	mw.dia = dia = jocrWindow()
	dia.show()
	return 0

#new item called test
action = QAction("test",mw)
#when activated run testopen
action.triggered.connect(lambda: testShow())
#add test to the Tools menu in Anki
mw.form.menuTools.addAction(action)