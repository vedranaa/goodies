"""
Run from command line as
vis3D https://qim.compute.dtu.dk/data-repository/InSegt_data/3D/nerves_part.tiff
or (assuming text file with the link exists)
vis3D /Users/vand/Documents/PROJECTS2/goodies/goodies_to_merge/vis3D/nerves_part_url_link.txt
or (assuming a folder with tif images exists)
vis3D /Users/vand/Documents/PROJECTS2/goodies/goodies_to_merge/testing_data/walnut
or (assuming a stacked tif file exists)
vis3D /Users/vand/Documents/PROJECTS2/goodies/goodies_to_merge/testing_data/nerves_part.tiff
or
vis3D and point to a supported file or folder.


Currently supported:
- folder tif* images
- .tif* file with stacked images
- url to tif* file
- .txt (or .link) file containing a url or file/folder path

TODO add support for: 
- folder with other image types readable with PIL
- dcm images (via pydicom)
- .nii.gz file, load it as numpy array via nibabel

TODO: Have basic version work with PIL and add optional imports of tifffile, 
nibabel and/or pydicom.
    
"""

import sys 
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui
import os
import glob
import tifffile
import PIL.Image
import numpy as np
import urllib.request

  
class Vis3D(PyQt5.QtWidgets.QWidget):
    
    def __init__(self, getslice, Z, z=None):
        
        super().__init__() 
        
        self.Z = Z
        if z==None:
            self.z = Z//2
        else:
            self.z = z
        self.getslice = getslice
            
        # Pixmap layers and atributes
        self.imagePix = self.grayToPixmap(self.getslice(self.z))
        self.zoomPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), self.imagePix.height()) 
        self.zoomPix.fill(self.transparentColor)
        
        # Atributes relating to the transformation between widget 
        # coordinate system and image coordinate system
        self.zoomFactor = 1 # accounts for resizing of the widget and for zooming in the part of the image
        self.padding = PyQt5.QtCore.QPoint(0, 0) # padding when aspect ratio of image and widget does not match
        self.target = PyQt5.QtCore.QRect(0, 0, self.width(),self.height()) # part of the target being drawn on
        self.source = PyQt5.QtCore.QRect(0, 0, 
                self.imagePix.width(), self.imagePix.height()) # part of the image being drawn
        self.offset = PyQt5.QtCore.QPoint(0, 0) # offset between image center and area of interest center
        
        # Atributes relating to zooming
        self.activelyZooming = False 
        self.newZoomValues = None
        self.clickedPoint = PyQt5.QtCore.QPoint()
        self.zoomPainter = None
        
        # Label for displaying text overlay
        self.textField = PyQt5.QtWidgets.QLabel(self)
        self.textField.setStyleSheet("background-color: rgba(191,191,191,191)")
        self.textField.setTextFormat(PyQt5.QtCore.Qt.RichText)
        self.textField.resize(0,0)
        self.textField.move(10,10)     
        self.hPressed = False
        self.textField.setAttribute(PyQt5.QtCore.Qt.WA_TransparentForMouseEvents)
        
        # Timer for displaying text overlay
        self.timer = PyQt5.QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hideText)

        # Playtime
        self.setTitle()
        initial_zoom = min(2000/max(self.imagePix.width(), 
                4*self.imagePix.height()/3),1) # downsize if larger than (2000,1500)
        self.resize(initial_zoom*self.imagePix.width(), 
                    initial_zoom*self.imagePix.height())
        self.showInfo('<i>Starting Vis3D</i> <br> For help, hit <b>H</b>', 5000)
        print("Starting Vis3D. For help, hit 'H'.")
    
    # constants
    transparentColor = PyQt5.QtGui.QColor(0, 0, 0, 0)    
    zoomColor = PyQt5.QtGui.QColor(0, 0, 0, 128) 
    
    helpText = (
        '<i>Help for Vis3DD</i> <br>' 
        '<b>KEYBOARD COMMANDS:</b> <br>' 
        '&nbsp; &nbsp; <b>H</b> shows this help <br>' 
        '&nbsp; &nbsp; <b>Arrow keys</b> change slice <br>' 
        '<b>MOUSE DRAG:</b> <br>' 
        '&nbsp; &nbsp; Zooms ')
    
    @classmethod
    def folderSlicer(cls, filename):
        # TODO: figure out dominant image format in the folder
        D = sorted(glob.glob(filename + '/*.tif*'))
        Z = len(D)
        readslice = lambda z: tifffile.imread(D[z])
        
        return readslice, Z

    @classmethod
    def tifVolSlicer(cls, filename):
        tif = tifffile.TiffFile(filename)
        Z = len(tif.pages)
        readslice = lambda z: tifffile.imread(filename, key = z)
        
        return readslice, Z

    @classmethod
    def urlPathSlicer(cls, url):
        
        volfile = PIL.Image.open(urllib.request.urlopen(url))
        Z = volfile.n_frames

        def readslice(z):
            volfile.seek(z)
            return(np.array(volfile))
        
        return readslice, Z
    
    @classmethod  
    def volSlicer(cls, vol):
        Z = vol.shape[0]
        readslice = lambda z: vol[z]
        
        return readslice, Z
        
    def showHelp(self):
        self.timer.stop()
        self.showText(self.helpText)
    
    def showInfo(self, text, time=1000):
        if not self.hPressed:
            self.timer.start(time)
            self.showText(text)
    
    def showText(self, text):
        self.textField.setText(text)
        self.textField.adjustSize()
        self.update()
          
    def hideText(self):
        self.textField.resize(0,0)
        self.update()
        
    def setTitle(self):
        self.setWindowTitle(f'z={self.z}/{self.Z}')
        
    def makeZoomPainter(self):
        painter_scribble = PyQt5.QtGui.QPainter(self.zoomPix)       
        painter_scribble.translate(-self.offset)
        painter_scribble.scale(1/self.zoomFactor, 1/self.zoomFactor)
        painter_scribble.translate(-self.padding)        
        painter_scribble.setCompositionMode(
                    PyQt5.QtGui.QPainter.CompositionMode_Source)
        return painter_scribble
    
    def paintEvent(self, event):
        """ Paint event for displaying the content of the widget."""
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(
                    PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        painter_display.drawPixmap(self.target, self.imagePix, self.source)
        if self.activelyZooming:
            painter_display.drawPixmap(self.target, self.zoomPix, self.source)
            
    def mousePressEvent(self, event):
        if event.button() == PyQt5.QtCore.Qt.LeftButton: 
            self.activelyZooming = True
            self.clickedPoint = event.pos()
            self.zoomPix.fill(self.transparentColor) # clear (fill with transparent)
            self.zoomPainter = self.makeZoomPainter()          
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.activelyZooming: 
            self.zoomPix.fill(self.transparentColor) # clear (fill with transparent)
            x = min(self.clickedPoint.x(), event.x())
            y = min(self.clickedPoint.y(), event.y())
            w = abs(self.clickedPoint.x() - event.x())
            h = abs(self.clickedPoint.y() - event.y())
            self.zoomPainter.fillRect(x, y, w, h, self.zoomColor)       
            self.update()
    
    def mouseReleaseEvent(self, event):  
        x = min(self.clickedPoint.x(), event.x())
        y = min(self.clickedPoint.y(), event.y())
        w = abs(self.clickedPoint.x() - event.x())
        h = abs(self.clickedPoint.y() - event.y())
        if w>2 and h>2: # Not zooming to les than 2x2 pixels
            self.newZoomValues = PyQt5.QtCore.QRect(x,y,w,h)
            self.executeZoom()
        else: 
            self.resetZoom()
        self.zoomPainter = None
        self.activelyZooming = False   
        self.update()
            
    def resizeEvent(self, event):
        """ Triggered by resizing of the widget window. """
        self.adjustTarget()
                
    def adjustTarget(self):
        """ Computes padding needed such that aspect ratio of the image is correct. """
       
        zoomWidth = self.width()/self.source.width()
        zoomHeight = self.height()/self.source.height() 
        
        # depending on aspect ratios, either pad up and down, or left and rigth
        if zoomWidth > zoomHeight:
            self.zoomFactor = zoomHeight
            self.padding = PyQt5.QtCore.QPoint(int((self.width() 
                            - self.source.width()*self.zoomFactor)/2), 0)
        else:
            self.zoomFactor = zoomWidth
            self.padding = PyQt5.QtCore.QPoint(0, int((self.height()
                            - self.source.height()*self.zoomFactor)/2))
            
        self.target = PyQt5.QtCore.QRect(self.padding, 
                            self.rect().bottomRight() - self.padding)
                   
    def executeZoom(self):
        """ Zooms to rectangle given by newZoomValues. """
        self.newZoomValues.translate(-self.padding)
        self.source = PyQt5.QtCore.QRect(self.newZoomValues.topLeft()/self.zoomFactor,
                self.newZoomValues.size()/self.zoomFactor)
        self.source.translate(-self.offset)
        self.source = self.source.intersected(self.imagePix.rect()) 
        self.showInfo('Zooming to ' + self.formatQRect(self.source))     
        self.offset = self.imagePix.rect().topLeft() - self.source.topLeft()
        self.adjustTarget()
        self.newZoomValues = None
    
    def resetZoom(self):
        """ Back to original zoom """
        self.source = PyQt5.QtCore.QRect(0,0,self.imagePix.width(), 
                                         self.imagePix.height())
        self.showInfo('Reseting zoom to ' + self.formatQRect(self.source))        
        self.offset = PyQt5.QtCore.QPoint(0,0)
        self.adjustTarget()        
        self.newZoomValues = None
            
    def keyPressEvent(self, event):

        if event.key()==PyQt5.QtCore.Qt.Key_Up: # uparrow          
            self.z = min(self.z+1, self.Z-1)
            self.imagePix = self.grayToPixmap(self.getslice(self.z))
            self.update()

        elif event.key()==PyQt5.QtCore.Qt.Key_Down: # downarrow
            self.z = max(self.z-1, 0)
            self.imagePix = self.grayToPixmap(self.getslice(self.z))
            self.update()
            
        elif event.key()==PyQt5.QtCore.Qt.Key_Right: 
            self.z = min(self.z+10, self.Z-1)
            self.imagePix = self.grayToPixmap(self.getslice(self.z))
            self.update()

        elif event.key()==PyQt5.QtCore.Qt.Key_Left: 
            self.z = max(self.z-10, 0)
            self.imagePix = self.grayToPixmap(self.getslice(self.z))
            self.update()

        elif event.key()==PyQt5.QtCore.Qt.Key_H: # h        
            if not self.hPressed:
                self.hPressed = True
                self.showHelp()
        elif event.key()==PyQt5.QtCore.Qt.Key_Escape: # escape
            self.closeEvent(event)
        self.setTitle()
        
    def keyReleaseEvent(self, event):
        if event.key()==PyQt5.QtCore.Qt.Key_H: # h
            self.hideText()
            self.hPressed = False
            
    # def closeEvent(self, event):
    #     self.showInfo("Bye, I'm closing")
    #     PyQt5.QtWidgets.QApplication.quit()
    #     # hint from: https://stackoverflow.com/questions/54045134/pyqt5-gui-cant-be-close-from-spyder
    #     # should also check: https://github.com/spyder-ide/spyder/wiki/How-to-run-PyQt-applications-within-Spyder
   
    @staticmethod
    def grayToPixmap(gray):
        """Np grayscale array to Qt pixmap. Assumes an 8-bit grayscale image."""
        
        bytesPerLine = gray.nbytes//gray.shape[0]
        qimage = PyQt5.QtGui.QImage(gray.data, gray.shape[1], gray.shape[0],
                                    bytesPerLine,
                                    PyQt5.QtGui.QImage.Format_Grayscale8)
        qpixmap = PyQt5.QtGui.QPixmap(qimage)
        return qpixmap
    
    @staticmethod
    def formatQRect(rect):
        coords =  rect.getCoords()
        s = f'({coords[0]},{coords[1]})--({coords[2]},{coords[3]})'
        return(s)  
    
def chose_file():
    file_dialog = PyQt5.QtWidgets.QFileDialog()
    
    
    # TODO, file_dialog.setHistory
    # A text file with history may be placed in
    # pathlib.Path(__file__).resolve()
    
    if file_dialog.exec_(): # file chosen
        filenames = file_dialog.selectedFiles()
        filename = filenames[0]
        return filename
    # else retuns None
    
def resolve_mode(filename):
    mode = None
    if os.path.isdir(filename):
        mode = 'folder'
    elif ((len(filename)>4) and (filename[:4]=='http') and 
          ('tif' in os.path.splitext(filename)[-1])):
        mode = 'url'
    else:
        ext = os.path.splitext(filename)[-1]
        if 'tif' in ext:
            mode = 'tifvol'
        elif (ext=='.txt') or (ext=='.link'):
            with open(filename) as f:
                content = f.read().strip()
                filename, mode = resolve_mode(content)
    return filename, mode



def slicer(filename):
    
    app = PyQt5.QtWidgets.QApplication([]) 
    
    # Special case when passed 3D numpy array.
    if ((type(filename)==np.ndarray) and (filename.ndim==3)):
        readslice, Z = Vis3D.volSlicer(filename)
 
    # Normal case when given filename/foldername/url
    else:
        if not filename:
            filename = chose_file()
                
        filename, mode = resolve_mode(filename)

        if mode: # file type identified
            if mode=='tifvol':
                readslice, Z = Vis3D.tifVolSlicer(filename)
            elif mode=='folder':
                readslice, Z = Vis3D.folderSlicer(filename)
            elif mode=='url':
                readslice, Z = Vis3D.urlPathSlicer(filename)    
            
            try: # read one slice (the last one) before initiating vis3D
                readslice(Z-1)       
            except:
                raise Exception(f"Can't read volume slices from filename {filename}.")
        else:
            raise Exception(f'Mode not identified for fielname {filename}.')
            
    vis3D = Vis3D(readslice, Z)
    vis3D.show()
    app.exec() 
        
    
def main():
    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = None        
    
    slicer(filename)
    
         
    
if __name__ == '__main__':
    
    '''
    Vis3D may be used from command-line. If no filename  or url is given, 
    it may be chosen via en dialog window.
    '''
    main()

   
    
    
    
    