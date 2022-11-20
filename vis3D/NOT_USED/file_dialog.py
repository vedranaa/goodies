import sys
import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets

class filedialogdemo(PyQt5.QtWidgets.QWidget):

   def __init__(self):
      
      super().__init__() 
		
      layout = PyQt5.QtWidgets.QVBoxLayout()
      
      self.btn = PyQt5.QtWidgets.QPushButton("Choose an image")
      self.btn.clicked.connect(self.getfile)
      layout.addWidget(self.btn)
   
      self.le = PyQt5.QtWidgets.QLabel("Image placeholder")	
      layout.addWidget(self.le)
    
      self.setLayout(layout)
      self.setWindowTitle("File Dialog demo")
		
   def getfile(self):
      fname = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\', "Image files (*.jpg *.gif)")
      self.le.setPixmap(PyQt5.QtGui.QPixmap(fname[0]))	
				
def main():
   app = PyQt5.QtWidgets.QApplication(sys.argv)
   ex = filedialogdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()