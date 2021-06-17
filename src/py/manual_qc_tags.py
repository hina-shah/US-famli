from pathlib import Path
from PIL import Image
import numpy as np

import shutil
import sys

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QComboBox, QCheckBox, QShortcut
from PyQt5.QtGui import QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random
import pandas as pd
import csv

from FamliFileIO import taginfo


class App(QDialog):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.initData()
        self.initUI()

    def initData(self):
        self.tags_file_list =[]
        self.current_study_ind = -1
        self.current_ind = -1
        self.qced_study_list = []
        self.qc_file_name = 'QC_Study_list.txt'
        self.img_dir = None
        self.current_tag_mgr = None
        self.show_only_unknown = False
        with open('us_tags.txt', 'r') as f:
            self.tag_list = [t.strip() for t in f.readlines()]
        self.tag_list.append('Unknown')
        self.tag_list.append('Undecided')
        self.tag_list.append('No tag')

    def initUI(self):
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.label = QLabel('')
        # Just some button connected to `plot` method

        self.tag_combobox = QComboBox()
        self.tag_combobox.addItems(self.tag_list)

        self.kbutton = QPushButton("Change tag text")
        self.kbutton.clicked.connect(self.changeTagForCurrent)

        self.sbutton = QPushButton('Save')
        self.sbutton.clicked.connect(self.save)
        
        self.obutton = QPushButton('Open next study')
        self.obutton.clicked.connect(self.openNextStudy)
        
        self.dbutton = QPushButton('Select Directory with Tags')
        self.dbutton.clicked.connect(self.openDirDialog)
        
        self.pbutton = QPushButton('Show previous image')
        self.pbutton.clicked.connect(self.plotPrevIm)
        self.nextbutton = QPushButton('Show next image')
        self.nextbutton.clicked.connect(self.plotNextIm)
        
        self.uu_checkbox = QCheckBox("Show Only Unknown Cines")
        self.uu_checkbox.setChecked(self.show_only_unknown)
        self.uu_checkbox.stateChanged.connect(self.showOnlyUnknowns)

        self.shortcut_next = QShortcut(QKeySequence.MoveToNextChar, self)
        self.shortcut_next.activated.connect(self.plotNextIm)

        self.shortcut_prev = QShortcut(QKeySequence.MoveToPreviousChar, self)
        self.shortcut_prev.activated.connect(self.plotPrevIm)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.dbutton)
        layout.addWidget(self.obutton)
        layout.addWidget(self.uu_checkbox)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addWidget(self.tag_combobox)
        layout.addWidget(self.kbutton)
        layout.addWidget(self.sbutton)
        layout.addWidget(self.pbutton)
        layout.addWidget(self.nextbutton)

        self.setLayout(layout)
        self.show()
    
    def openDirDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dirname = QFileDialog.getExistingDirectory(self, "Choose the Image Directory")
        print(dirname)
        if dirname:
            self.img_dir = Path(dirname)
            if not self.img_dir.exists():
                self.img_dir.resolve()
                print('{} does not exist'.format(self.img_dir))
                return
            # Find the list of tags in the directory
            self.tags_file_list = list(self.img_dir.glob("**/"+ taginfo.TagInfoFile.file_name))
            QMessageBox.information(self, "", "Found a list of {} studies".format(len(self.tags_file_list)))

            self.qced_study_list = []
            qc_file_path = self.img_dir/self.qc_file_name
            if qc_file_path.exists():
                with open(qc_file_path, 'r') as f:
                    self.qced_study_list = [r.strip() for r in f.readlines()]
            if len(self.qced_study_list) > 0:
                cleaned_studies = [ tag_file for tag_file in self.tags_file_list if tag_file.parent.name not in self.qced_study_list ]
                self.tags_file_list = cleaned_studies
            QMessageBox.information(self, "", "Will process a list of {} studies (removing already QCed)".format(len(self.tags_file_list)))

            if self.img_dir and len(self.tags_file_list) > 0:
                self.current_study_ind = -1
                self.openNextStudy()

    def openNextStudy(self):
        # if current study index is >=0, then save the current state
        self.save()
        self.current_tag_mgr = None
        print('Set current tag mgr to nothing')

        self.current_study_ind +=1
        tag_file = self.tags_file_list[self.current_ind]
        print("tag file: {}".format(tag_file))
        QMessageBox.information(self, "", "Processing study: {}".format(tag_file.parent.name))
        self.current_tag_mgr = taginfo.TagInfoFile(tag_file.parent)
        self.current_tag_mgr.read()
        # get a copy of the previously created tags for all files
        self.input_im_list = self.current_tag_mgr.getAllRows()
        # Start processing it.
        self.startProcessing()

    def startProcessing(self):
        
        if self.img_dir and len(self.tags_file_list) > 0 and len(self.input_im_list) > 0:
            self.current_ind = -1
            self.plotNextIm()
        else:
            QMessageBox.warning(self, "", "Didn't find any images in the directory, chose a directory with images")

    def getimname(self):
        if self.current_tag_mgr is None:
            return None

        tag_dir = self.current_tag_mgr.getStudyDir()

        print(self.input_im_list[self.current_ind])
        file_in = Path(self.input_im_list[self.current_ind]['File'])
        img_name = file_in.name.replace(file_in.suffix, '.jpg')

        img_path = tag_dir / img_name
        return img_path

    def getNextInd(self, forward):

        if self.current_tag_mgr is None:
            return

        increment = 1 if forward else -1
        self.current_ind += increment
        if self.show_only_unknown:
            while 0 <= self.current_ind and self.current_ind < len(self.input_im_list):
                r = self.input_im_list[self.current_ind]
                if r['tag'] in ['Unknown', 'Undecided', 'No tag'] and r['type'] == 'cine':
                    break
                else:
                    self.current_ind += increment

    def plotPrevIm(self):

        if len(self.input_im_list) == 0 or self.img_dir is None:
            QMessageBox.warning(self, "", "Either the image list is empty or directory not selected, please make sure both are selected")
            print(len(self.input_im_list))
            print(self.img_dir)
            return

        if self.current_ind == 0:
            QMessageBox.information(self, "", "Reached the first Image")
            return
        self.changeTagForCurrent()
        self.getNextInd(forward=False)
        if self.current_ind >=0 and self.current_ind < len(self.input_im_list):
            img_path = self.getimname()
            if img_path.exists():
                self.plot(img_path)
            else:
                QMessageBox.information(self, "", "Didn't find image: {}, moving to next".format(img_path))
                self.plotPrevIm()

    def plotNextIm(self):

        if len(self.input_im_list) == 0 or self.img_dir is None:
            QMessageBox.warning(self, "", "Either the image list is empty or directory not selected, please make sure both are selected")
            print(len(self.input_im_list))
            print(self.img_dir)
            return

        if self.current_ind == len(self.input_im_list):
            QMessageBox.information(self, "", "Done with this study")
            return

        self.changeTagForCurrent()
        self.getNextInd(forward=True)
        if self.current_ind >=0 and self.current_ind < len(self.input_im_list):
            img_path = self.getimname()
            if img_path.exists():
                self.plot(img_path)
            else:
                QMessageBox.information(self, "", "Didn't find image: {}, moving to next".format(img_path))
                self.plotNextIm()

    def save(self):
        if self.current_tag_mgr is not None and self.current_ind >= 0:
            orig_file = self.current_tag_mgr.getFilePath()
            target_file = self.current_tag_mgr.getStudyDir() / taginfo.TagInfoFile.file_name.replace('.csv', '_old.csv')
            if not target_file.exists():
                shutil.copyfile(orig_file, target_file)

            self.current_tag_mgr.write()
            study_name = self.tags_file_list[self.current_ind].parent.name
            if study_name not in self.qced_study_list:
                self.qced_study_list.append(study_name)
                with open(self.img_dir/self.qc_file_name, 'a+') as f:
                    f.write(study_name + "\n")
            print('Wrote to qc file')

    def plot(self, img_path):
        # Display the image
        
        img = Image.open(str(img_path))
        nda = np.asarray(img)
        # instead of ax.hold(False)
        plt.imshow(nda)

        # refresh canvas
        self.canvas.draw()
        r = self.input_im_list[self.current_ind]

        self.label.setText("::: {}/{} ::: Study: {}, Image Name: {}, Type: {}, Tag: {}".format( self.current_ind+1, len(self.input_im_list), self.current_tag_mgr.getStudyName(), img_path.name, r['type'], r['tag']))

        if r['tag'] in self.tag_list:
            tag_index = self.tag_list.index(r['tag'])
        else:
            tag_index = self.tag_list.index('Unknown')
        self.tag_combobox.setCurrentIndex(tag_index)
        self.update()
    
    def changeTagForCurrent(self):
        if self.current_tag_mgr is not None and self.current_ind >=0:
            r = self.input_im_list[self.current_ind]
            new_tag = self.tag_combobox.currentText()
            if r['tag'] != new_tag:
                print("Changing tag from {} to {}".format(r['tag'], new_tag))
                self.input_im_list[self.current_ind]['tag'] = new_tag

    def showOnlyUnknowns(self):
        self.show_only_unknown = self.uu_checkbox.isChecked()
        if self.show_only_unknown and self.current_tag_mgr is not None:
            current_tag = self.input_im_list[self.current_ind]['tag']
            if current_tag not in ['Undecided', 'Unknown', 'No tag']:
                self.plotNextIm()

    def closeEvent(self,event):
        result = QMessageBox.question(self,
                      "Confirm Exit...",
                      "Are you sure you want to exit ?",
                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()
        if result == QMessageBox.Yes:
            self.save()
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = App()
    
    sys.exit(app.exec_())