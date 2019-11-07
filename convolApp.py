from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog
import sys
import UI.convolution as convolution
import numpy as np
import cv2
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class convolApp(QtWidgets.QDialog , convolution.Ui_Dialog):
    def __init__(self, parant=None):
        super(convolApp, self).__init__(parant)
        self.setupUi(self)

        self.btn11.clicked.connect(self.btn11_Clicked)
        self.btn12.clicked.connect(self.btn12_Clicked)
        self.btn13.clicked.connect(self.btn13_Clicked)
        self.btn14.clicked.connect(self.btn14_Clicked)
        self.btn21.clicked.connect(self.btn21_Clicked)
        self.btn22.clicked.connect(self.btn22_Clicked)
        self.btn31.clicked.connect(self.btn31_Clicked)
        self.btn32.clicked.connect(self.btn32_Clicked)
        self.count = 0
        self.pts1 = []
        self.pts2 = np.float32([[20,20],[450,20],[450,450],[20,450]])
        self.btn41.clicked.connect(self.btn41_Clicked)
        self.btn42.clicked.connect(self.btn42_Clicked)
        self.btn43.clicked.connect(self.btn43_Clicked)
        self.btn44.clicked.connect(self.btn44_Clicked)

    def btn11_Clicked(self):
        image = cv2.imread("res/dog.bmp")
        cv2.imshow("Image",image)

    def btn12_Clicked(self):
        """
        opencv 的三通道依次為 b g r
        ex:
        b, g, r = cv2.split(image)
        cv2.imshow("B",b) 0
        cv2.imshow("G",g) 1
        cv2.imshow("R",r) 2
        """
        image = cv2.imread("res/color.png")
        image = image[:,:,[1,2,0]]
        cv2.imshow("Image",image)

    def btn13_Clicked(self):
        image = cv2.imread("res/dog.bmp")
        flip_img = cv2.flip(image,1) # image[height, witdh, channel]
        cv2.imshow("Non Flipped", image)
        cv2.imshow("Flipped", flip_img)

    def btn14_Clicked(self):
        cv2.namedWindow("Blending",1)
        cv2.createTrackbar("Blend","Blending",0,255,self.Blending_func)
        self.Blending_func("Blend")

    def Blending_func(self,x):
        image = cv2.imread("res/dog.bmp")
        flip_img = cv2.flip(image,1) # image[height, witdh, channel]
        alpha = cv2.getTrackbarPos("Blend","Blending")/255.0
        beta = 1-alpha
        dst = cv2.addWeighted(image,alpha,flip_img,beta,0)
        cv2.imshow("Blending", dst)

    def btn21_Clicked(self):
        image = cv2.imread("res/QR.png",cv2.IMREAD_GRAYSCALE)
        th ,dst = cv2.threshold(image,80,255,cv2.THRESH_BINARY)
        cv2.imshow("Before", image)
        cv2.imshow("After", dst)
        plt.show()

    def btn22_Clicked(self):
        image = cv2.imread("res/QR.png",cv2.IMREAD_GRAYSCALE)
        dst = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,-1)
        cv2.imshow("Before", image)
        cv2.imshow("After", dst)

    def btn31_Clicked(self):
        image = cv2.imread("res/OriginalTransform.png")
        # TODO : translation
        rows,cols,channel = image.shape
        tx = int(self.lineEdit_3.text())
        ty = int(self.lineEdit_4.text())
        M = np.float32([[1,0,tx], [0,1,ty]])
        dst = cv2.warpAffine(image,M,(cols,rows))

        # TODO : rotation & scaling
        center = (130+tx,125+ty)
        angle = float(self.lineEdit_1.text())
        scale = float(self.lineEdit_2.text())
        M = cv2.getRotationMatrix2D(center, angle,scale)
        dst = cv2.warpAffine(dst,M,(cols,rows))
        cv2.imshow('Before', image)
        cv2.imshow('Transformation', dst)

    def btn32_Clicked(self):
        image = cv2.imread("res/OriginalPerspective.png")
        cv2.namedWindow('OriginalPerspective', 0)
        cv2.imshow('OriginalPerspective', image)
        cv2.setMouseCallback('OriginalPerspective', self.get_point)

    def get_point(self,event,x,y,flag,param):
        image = cv2.imread("res/OriginalPerspective.png")
        if event == cv2.EVENT_LBUTTONDOWN:
            self.count += 1
            if self.count == 1:
                self.pts1 = np.float32([[x,y]])
            else :
                temp = np.float32([[x,y]])
                self.pts1 = np.vstack((self.pts1, temp))
            if self.count == 4:
                M = cv2.getPerspectiveTransform(self.pts1,self.pts2)
                self.pts1 = np.float32([])
                self.count = 0
                rows,cols,channel = image.shape
                dst = cv2.warpPerspective(image,M,(450,450))
                cv2.imshow("Result",dst)
                
            
    def btn41_Clicked(self):
        # 3x3 Gaussian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel =np.exp(-(x**2+y**2))

        # Normaliztion
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        image = mpimg.imread('res/School.jpg')
        gray = self.rgb2gray(image)
        dst = signal.convolve2d(gray ,gaussian_kernel,boundary = 'symm', mode = 'same')
        plt.imshow(dst, cmap=plt.get_cmap('gray'))

        plt.axis('off')
        plt.savefig('res/gaussin.jpg')
        plt.show()

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    # TODO : Horizontal Sobel edge
    def btn42_Clicked(self):
        filter_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        image = mpimg.imread('res/gaussin.jpg')
        gray = self.rgb2gray(image)
        dst = signal.convolve2d(gray ,filter_x, boundary = 'symm', mode = 'same')
        dst = np.sqrt(np.square(dst))
        dst *= 255.0 / dst.max()
        plt.imshow(dst ,cmap = 'gray')
        plt.title("Horizontal")
        plt.axis('off')
        plt.show()

    # TODO : Vertical Sobel edge
    def btn43_Clicked(self):
        filter_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        image = mpimg.imread('res/gaussin.jpg')
        gray = self.rgb2gray(image)
        dst = signal.convolve2d(gray ,filter_y, boundary = 'symm', mode = 'same')
        dst = np.sqrt(np.square(dst))
        dst *= 255.0 / dst.max()
        plt.imshow(dst ,cmap = 'gray')
        plt.title("Vertical")
        plt.axis('off')
        plt.show()

    def btn44_Clicked(self):
        filter_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        image = mpimg.imread('res/gaussin.jpg')
        gray = self.rgb2gray(image)
        dst_x = signal.convolve2d(gray ,filter_x, boundary = 'symm', mode = 'same')
        
        filter_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        image = mpimg.imread('res/gaussin.jpg')
        gray = self.rgb2gray(image)
        dst_y = signal.convolve2d(gray ,filter_y, boundary = 'symm', mode = 'same')
       
        magnitude = np.sqrt(np.square(dst_x)+np.square(dst_y))
        magnitude *= 255.0/magnitude.max()
        plt.imshow(magnitude ,cmap = 'gray')
        plt.axis('off')
        plt.show()
def main():
    app = QApplication(sys.argv)
    form = convolApp()
    form.show()
    app.exec_()

if __name__ == '__main__' :
    main()