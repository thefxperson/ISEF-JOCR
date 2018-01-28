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
		
		self.canvas = canvas(self)
		self.layout.addWidget(self.canvas,1,1)

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


def canvas(QWidget):				#wip
	def __init__(self):
		super(canvas,self).__init__()

		self.setFixedSize(200,200)

	def paintEvent(self, event):
		painter = QPainter()
		painter.begin(self)

		self.drawPoints(painter)

		painter.end()

	def mousePressEvent(self, event):
		print("pressed:", event.button())
		self.update()

	def mouseReleasedEvent(self, event):
		print("released:", event.button())
		self.update()

	def mouseMovedEvent(self, event):
		print(event.x(),",",event.y())
		self.update()

	def drawPoints(self, qp):
		qp.setPen(Qt.red)
		size = self.size()

		for i in range(1000):
			x = random.randint(1, size.width()-1)
			y = random.randint(1, size.height()-1)
			qp.drawPoint(x, y)     

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