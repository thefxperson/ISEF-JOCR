from aqt import mw
from aqt.utils import showInfo
from aqt.qt import *

class jocrWindow(QWidget):
	def __init__(self, numChars):
		super(jocrWindow, self).__init__()

		self.setGeometry(300,300,400,400)	#x,y on screen, width and height
		self.setWindowTitle("jocr")
		self.canvasList = [MyCanvasPainter() for x in range(numChars)]
		self.currentWindow = 0
		self.numChars = numChars


		#------------------GUI----------------

		self.layout = QGridLayout()

		self.saver = QPushButton("Evaluate")
		self.layout.addWidget(self.saver, 0, 0)

		self.top3 = QLabel("top 3: A B C")
		self.layout.addWidget(self.top3,0,1,1,4)

		self.clear = QPushButton("Clear",self)
		self.layout.addWidget(self.clear,0,5)

		#-------------------------------------

		self.left = QPushButton("<-",self)
		self.left.setMinimumHeight(300)
		self.layout.addWidget(self.left,1,0)
		
		self.canvas = self.currentCanvas()#MyCanvas()
		self.layout.addWidget(self.canvas,1,1,1,4)

		self.right = QPushButton("->",self)
		self.right.setMinimumHeight(300)
		self.layout.addWidget(self.right,1,5)

		#--------------------------------------

		self.strokes = QLabel(str(self.currentWindow))
		self.layout.addWidget(self.strokes,2,0)

		self.override = QLabel("Override:")
		self.layout.addWidget(self.override,2,1)

		self.red = QPushButton("<10m",self)
		self.layout.addWidget(self.red,2,2)

		self.green = QPushButton("10m",self)
		self.layout.addWidget(self.green,2,3)

		self.blue = QPushButton("3d",self)
		self.layout.addWidget(self.blue,2,4)

		self.undo = QPushButton("Undo", self)
		self.layout.addWidget(self.undo,2,5)

		#-------------------------------------
		self.clear.clicked.connect(self.clearOne)
		self.undo.clicked.connect(self.undoOne)
		self.left.clicked.connect(self.leftCan)
		self.right.clicked.connect(self.rightCan)
		self.saver.clicked.connect(self.eval)

		self.layout.setRowStretch(1,1)

		self.setLayout(self.layout)


	def currentCanvas(self):
		return self.canvasList[self.currentWindow]

	def leftCan(self):
		self.currentWindow -= 1
		if self.currentWindow < 0:
			self.currentWindow = self.numChars-1
		self.updateWindow()

	def rightCan(self):
		self.currentWindow += 1
		if self.currentWindow >= self.numChars:
			self.currentWindow = 0
		self.updateWindow()
	
	def updateWindow(self):
		self.layout.removeWidget(self.canvas)
		self.canvas.setParent(None)
		self.canvas = self.currentCanvas()
		self.layout.addWidget(self.canvas,1,1,1,4)

		self.layout.removeWidget(self.strokes)
		self.strokes.setParent(None)
		self.strokes = QLabel(str(self.currentWindow))
		self.layout.addWidget(self.strokes,2,0)

	def clearOne(self):
		canvas = self.currentCanvas()
		canvas.clearCanvas()

	def undoOne(self):
		canvas = self.currentCanvas()
		canvas.undoStroke()

	def eval(self):
		cwd = os.getcwd()
		path = cwd + "/nn/"
		if not os.path.exists(path):
			os.makedirs(path)
		canvas = self.currentCanvas()
		p = QPixmap.grabWindow(canvas.winId())
		if self.currentWindow < 10:
			t = "0" + str(self.currentWindow)
		else:
			t = str(self.currentWindow)
		p.save(path+"img"+t+".jpg", "jpg")


class MyCanvasPainter(QWidget):
	def __init__(self):
		super(MyCanvasPainter, self).__init__()

		self.setFixedSize(315, 315)

		self.qp = QPainter()
		self.strokes = []

	def paintEvent(self, event):
		self.qp.begin(self)

		self.qp.fillRect(event.rect(), QBrush(QColor(255,255,255)))
		for i in range(len(self.strokes)):
			self.qp.drawPath(self.strokes[i])

		self.qp.end()

	def mousePressEvent(self, event):
		if event.button() == 1:
			self.moving = True
			path = QPainterPath()
			self.strokes.append(path)
			self.strokes[len(self.strokes)-1].moveTo(event.x(), event.y())
		self.update()

	def mouseMoveEvent(self, event):
		if self.moving:
			self.strokes[len(self.strokes)-1].lineTo(event.x(), event.y())
		self.update()

	def mouseReleaseEvent(self, event):
		if event.button() == 1:
			self.moving = False
			self.strokes[len(self.strokes)-1].lineTo(event.x(), event.y())
		self.update()

	def clearCanvas(self):
		self.strokes = []
		self.update()

	def undoStroke(self):
		self.strokes = self.strokes[:-1]
		self.update()


def testShow():
	'''card = mw.col.sched._getCard()
	if not card:
		showInfo("deck finished")
	note = card.note()
	for (name, value) in note.items():
		if name == "Vocabulary-Kanji":
			numCanv = len(value)
		if name == "Vocabulary-English":
			eng = value'''
	#showInfo(card.q())
	#showInfo(card.a())
	#showInfo(str(numCanv))
	#showInfo(eng)
	mw.dia = dia = jocrWindow(14)
	dia.show()
	return 0

def create():
	showInfo("called")
	#new item called test
	action = QAction("JOCR",mw)
	#when activated run testopen
	action.triggered.connect(lambda: testShow())
	#add test to the Tools menu in Anki
	mw.form.menuTools.addAction(action)

#new item called test
action = QAction("JOCR",mw)
#when activated run testopen
action.triggered.connect(lambda: testShow())
#add test to the Tools menu in Anki
mw.form.menuTools.addAction(action)
