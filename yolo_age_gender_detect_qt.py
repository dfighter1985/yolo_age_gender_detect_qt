'''
Age and gender detection from face image using Yolov8

Copyright (C) 2024 dfighter1985

Installation
=============
python -mvenv qt
cd qt
Scripts\activate.bat
pip install pyqt6
pip install ultralytics

Copy the Yolo age and gender detection model to this directory with name best.pt.

Usage
======
python yolo_age_gender_detect_qt.py
Click browse and select a cropped face image.

NOTE: Model is not included. See the following blog post for dataset:
https://dfighter1985.wordpress.com/2024/05/20/converting-the-utkface-computer-vision-dataset-to-the-yolo-format/

'''
import sys
import os

from ultralytics import YOLO

from PyQt6.QtGui import QPixmap, QColor, QColorConstants

from PyQt6.QtWidgets import ( 
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QLineEdit,
    QFileDialog
)

MODEL_FILE = "best.pt"

class AgeGenderDetectionWidget( QWidget ):
    def __init__( self, yoloModel ):
        super().__init__( parent=None )
        self.model = yoloModel
        self.setWindowTitle("PyQt age / gender detection")
        
        self.emptyPixmap = QPixmap( 640, 480 )
        self.emptyPixmap.fill( QColorConstants.Gray )
        
        layout = QVBoxLayout()
        
        self.browseButton = QPushButton( "Browse" )
        self.lineEdit = QLineEdit()
        self.lineEdit.setEnabled( False )
        self.pixLabel = QLabel()
        
        self.pixLabel.setPixmap( self.emptyPixmap )
        
        hlayout = QHBoxLayout()
        hlayout.addWidget( self.browseButton )
        hlayout.addWidget( self.lineEdit )
        layout.addLayout( hlayout )
        layout.addWidget( self.pixLabel )
        
        self.genderLineEdit = QLineEdit( "N/A" )
        self.genderLineEdit.setEnabled( False )
        self.ageLineEdit = QLineEdit( "N/A" )
        self.ageLineEdit.setEnabled( False )
        self.confidenceLineEdit = QLineEdit( "N/A" )
        self.confidenceLineEdit.setEnabled( False )
        
        flayout = QFormLayout()
        flayout.addRow( "Gender", self.genderLineEdit )
        flayout.addRow( "Age", self.ageLineEdit )
        flayout.addRow( "Confidence", self.confidenceLineEdit )
        layout.addLayout( flayout )
        
        self.setLayout( layout )
        
        self.browseButton.clicked.connect( self.onBrowseClicked )
        
    def processResult( self, result ):
        # We should have a single bounding box, as we're processing a single face
        box = result.boxes[ 0 ]
        
        className = str( result.names[ int( box.cls.item() ) ] )
        confidence = str( box.conf.item() * 100.0 )
        
        print( "Prediction result:" )
        print( "Detected class: " + className + ", Detection confidence: " + confidence )
        
        age = ''
        gender = ''
        
        parts = className.split()
        if parts[ 0 ] == "adult" or parts[ 0 ] == 'senior':
            age = parts[ 0 ]
            gender = parts[ 1 ]
        else:
            age = parts[ 1 ]
            gender = parts[ 0 ]
            
        self.ageLineEdit.setText( age )
        self.genderLineEdit.setText( gender )
        self.confidenceLineEdit.setText( confidence )
        
    def onBrowseClicked( self ):
        file, filter = QFileDialog.getOpenFileName( None, "Test", os.getcwd(), "JPG files (*.jpg)" )
        
        print( "Selected file: '" + file + "'")
        
        if file != '':
            self.lineEdit.setText( file )
            pixMap = QPixmap( file )
            pixMap = pixMap.scaledToWidth( 640 )
            pixMap = pixMap.scaledToHeight( 480 )
            self.pixLabel.setPixmap( pixMap )
            
            print( "Detecting on image " + file + "..." )
            results = self.model.predict( file )
            print( "Done." )

            # We should have only one result, as we're processing a single face
            self.processResult( results[ 0 ] )

        else:
            self.pixLabel.setPixmap( self.emptyPixmap )
            self.ageLineEdit.setText( "N/A" )
            self.genderLineEdit.setText( "N/A" )
            self.confidenceLineEdit.setText( "N/A" )

if __name__ == "__main__":
    print( "Loading model " + MODEL_FILE + "..." )
    yoloModel = YOLO( MODEL_FILE )
    print( "Done." )
    
    print( "Starting application..." )
    app = QApplication( [] )
    
    print( "Done." )
    
    widget = AgeGenderDetectionWidget( yoloModel )
    widget.show()
    
    result = app.exec()
    
    print( "Bye!" )
    
    sys.exit( result )
