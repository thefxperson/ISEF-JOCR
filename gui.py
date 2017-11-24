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

		self.clear = QPushButton("Clear",self)
		self.layout.addWidget(self.clear,0,5)

		#-------------------------------------

		self.left = QPushButton("<-",self)
		self.left.setMinimumHeight(300)
		self.layout.addWidget(self.left,1,0)
		
		self.canvas = MyCanvas()
		self.layout.addWidget(self.canvas,1,1,1,4)
		self.canvas.selection_shape = "rect"

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

		#-------------------------------------
		self.clear.clicked.connect(self.canvas.clearCanvas)

		self.layout.setRowStretch(1,1)

		self.setLayout(self.layout)

class MyCanvas(QWidget):
	
	def __init__(self):
		super(MyCanvas, self).__init__()

		self.setFixedSize(315, 315)

		self.selection_shape = 'rect'

		self.selection = False

		self.shapes = []
		
	def paintEvent(self, event):

		self.qp = QPainter()

		pen = QPen()
		pen.setColor(QColor(0,255,0))
		pen.setWidth(5)
		self.qp.setPen(pen)

		self.qp.begin(self)

		self.qp.fillRect(event.rect(), QBrush(QColor(255,255,255)))

		for name, args in self.shapes:
			self.drawRect(event, self.qp, args)
				
		self.qp.end()
		
	def drawRect(self, event, qp, args):
		qp.drawRect(args['x'], args['y'], args['width'], args['height'])

	def mousePressEvent(self, event):
		if event.button() == 1:
			self.selection = True
			self.shapes.append(('rect', {'x':event.x(), 'y':event.y(), 'width':1, 'height':1}))
			
	def mouseReleaseEvent(self, event):
		if event.button() == 1 and self.selection:
			self.selection = False
			self.shapes.append(('rect', {'x':event.x(), 'y':event.y(), 'width':1, 'height':1}))
			
		self.update()


	def mouseMoveEvent(self, event):
		if self.selection:
			self.shapes.append(('rect', {'x':event.x(), 'y':event.y(), 'width':1, 'height':1}))
		self.update()

	def clearCanvas(self):
		del self.shapes[:]
		self.update()

def testShow():
	mw.dia = dia = jocrWindow()
	dia.show()
	return 0

#new item called test
action = QAction("JOCR",mw)
#when activated run testopen
action.triggered.connect(lambda: testShow())
#add test to the Tools menu in Anki
mw.form.menuTools.addAction(action)