import PyQt5.QtWidgets
   
app = PyQt5.QtWidgets.QApplication([]) # application has to be started before any widgets

file_dialog = PyQt5.QtWidgets.QFileDialog()
file_dialog.setDirectory('/Users/vand/Documents/PROJECTS2/goodies/goodies/testing_data')
 
if file_dialog.exec_():
   filenames = file_dialog.selectedFiles()
   print(filenames[0])
	
