# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:33:17 2020

@author: hatam
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2 
import math
#function
#caculate FFT of image
def DFTfilter(mat):
    mat_fft=cv2.dft(np.float32(mat),flags = cv2.DFT_COMPLEX_OUTPUT)
    shifted_mat=np.fft.fftshift(mat_fft)
    amp_mat=np.abs(cv2.magnitude(shifted_mat[:,:,0],shifted_mat[:,:,1]))
    log_amp_mat=15*np.log(amp_mat+np.ones(amp_mat.shape))
    return log_amp_mat
def GaussianFilter(nRows, nCols, sigma, highPass=True):
    if(nRows % 2 == 0):
       centerI =int(nRows/2)
    else:
       centerI =int(nRows/2)+1
    if(nCols % 2 == 0):
       centerJ =int(nCols/2)
    else:
       centerJ =int(nCols/2)+1
    filter_gauss=np.zeros((nRows,nCols))
    for j in range(nCols) :
     for i in range(nRows):
       g = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
       if(highPass):
           filter_gauss[i,j]=1-g
       else:
           filter_gauss[i,j]=g

    return filter_gauss
def cutoff(image , gussian,cut,lowPass=True ):
    imageDFT = fftshift(fft2(image))
    GussianDFT=gussian
    if(len(gussian) == 0):
       centerI =int(len(gussian)/2)
    else:
       centerI =int(len(gussian)/2)+1
    if(len(gussian[0]) % 2 == 0):
       centerJ =int(len(gussian[0])/2)
    else:
       centerJ =int(len(gussian[0])/2)+1
    cut_off=np.zeros_like(gussian)
    for i in range (-cut,cut,1):
        for j in range (-cut,cut,1):
              if(i**2+j**2<cut**2):
                cut_off[centerI+i,centerJ+j]=1 
    if(lowPass):
        GussianDFT_cut=GussianDFT * (cut_off)
    else:
        GussianDFT_cut=GussianDFT * (np.logical_not(cut_off))  
    imageFiltered=GussianDFT_cut*imageDFT
    return GussianDFT_cut, imageFiltered,ifft2(ifftshift(imageFiltered))
#read image
image1 = cv2.imread('obama.jpg')
image2 = cv2.imread('shakira.jpg')
#change size
image1_resized=np.delete(image1,np.s_[0:28:1],axis=0)
image1_resized=np.delete(image1_resized,np.s_[503:531:1],axis=0)
image1_resized=np.delete(image1_resized,np.s_[::9],axis=0)

image1_resized=np.delete(image1_resized,np.s_[0:50:1],axis=1)
image1_resized=np.delete(image1_resized,np.s_[692:744:1],axis=1)
image1_resized=np.delete(image1_resized,np.s_[::6],axis=1)

# now find  matching eye
#image1_resized[175:185,245:255,:]=(0,0,255)
#image2[185:195,245:255,:]=(0,0,255)
cv2.imwrite('q4_03_near.jpg',image1_resized)
cv2.imwrite('q4_04_far.jpg',image2)

#DFT of image

dft_image1_B=DFTfilter(image1_resized[:,:,0])
dft_image1_B=dft_image1_B.astype('uint8') 
cv2.imwrite('q4_05_dft_near.jpg',dft_image1_B)

dft_image2_B=DFTfilter(image2[:,:,0])
dft_image2_G=DFTfilter(image2[:,:,1])
dft_image2_R=DFTfilter(image2[:,:,2])
dft_image2_B=dft_image2_B.astype('uint8') 
#dft_image2=np.zeros_like(image2)
#dft_image2[:,:,0]=dft_image2_B
cv2.imwrite('q4_06_dft_far.jpg',dft_image2_B)

#Gussian
highpass=GaussianFilter(len(image1_resized), len(image1_resized[0]), 35, highPass=True)

lowpass=GaussianFilter(len(image2), len(image2[0]),8, highPass=False)


cv2.imwrite('Q4_07_highpass_35.jpg',80*np.real(np.log(np.absolute(highpass) + np.ones(highpass.shape))))
cv2.imwrite('Q4_08_lowpass_8.jpg',80*np.real(np.log(np.absolute(lowpass) + np.ones(lowpass.shape))))


lowpass_cutoff ,imagecut_lowpass_B,imageDFTlow_B=cutoff(image2[:,:,0] , lowpass,15,lowPass=True )
highpass_cutoff ,imagecut_highpass_B,imageDFThigh_B=cutoff(image1_resized[:,:,0] , highpass,10,lowPass=False )
if(len(highpass)%2 == 0):
       centerI =int(len(highpass)/2)
else:
       centerI =int(len(highpass)/2)+1
if(len(highpass[0]) % 2 == 0):
       centerJ =int(len(highpass[0])/2)
else:
       centerJ =int(len(highpass[0])/2)+1
cut_off=np.ones_like(highpass)
for i in range (-15,15,1):
        for j in range (-15,15,1):
              if(i**2+j**2<15**2 and i**2+j**2>10**2):
                cut_off[centerI+i,centerJ+j]=1/2
   
cv2.imwrite("Q4_10_lowpass_cutoff.jpg",80*np.real(np.log(np.absolute(lowpass_cutoff) + np.ones(lowpass_cutoff.shape))))
cv2.imwrite("Q4_09_highpass_cutoff.jpg",80*np.real(np.log(np.absolute(highpass_cutoff) + np.ones(highpass_cutoff.shape))))

cv2.imwrite("Q4_12_lowpassed.jpg",30*np.real(np.log(np.absolute(imagecut_lowpass_B) + np.ones(imagecut_lowpass_B.shape))))
cv2.imwrite("Q4_11_highpassed.jpg",30*np.real(np.log(np.absolute(imagecut_highpass_B) + np.ones(imagecut_highpass_B.shape))))

hybrid_im_freq_B=imagecut_highpass_B+imagecut_lowpass_B

cv2.imwrite("Q4_13_hybrid_frequency.jpg",30*np.real(np.log10(np.absolute(hybrid_im_freq_B) + np.ones(hybrid_im_freq_B.shape))))
hybrid_im_freq_B=np.multiply(hybrid_im_freq_B,cut_off)
hybrid_im_B=ifft2(ifftshift(hybrid_im_freq_B)) 
hybrid_im_B=np.real(hybrid_im_B)

#cv2.imwrite("high.jpg",np.real(imageDFThigh_B))



hybrid_im_freq_B=np.multiply(hybrid_im_freq_B,cut_off)


#channel green
lowpass_cutoff ,imagecut_lowpass_G,imageDFTlow_G=cutoff(image2[:,:,1] , lowpass,15,lowPass=True )
highpass_cutoff ,imagecut_highpass_G,imageDFThigh_G=cutoff(image1_resized[:,:,1] , highpass,10,lowPass=False )

hybrid_im_freq_G=imagecut_highpass_G+imagecut_lowpass_G
hybrid_im_freq_G=np.multiply(hybrid_im_freq_G,cut_off)
hybrid_im_G=ifft2(ifftshift(hybrid_im_freq_G)) 
hybrid_im_G=np.real(hybrid_im_G)
hybrid_im_G=hybrid_im_G.astype('uint8')

#channel red
lowpass_cutoff ,imagecut_lowpass_R,imageDFTlow_R=cutoff(image2[:,:,2] , lowpass,15,lowPass=True )
highpass_cutoff ,imagecut_highpass_R,imageDFThigh_R=cutoff(image1_resized[:,:,2] , highpass,10,lowPass=False )
hybrid_im_freq_R=imagecut_highpass_R+imagecut_lowpass_R
hybrid_im_freq_R=np.multiply(hybrid_im_freq_R,cut_off)
hybrid_im_R=ifft2(ifftshift(hybrid_im_freq_R)) 
hybrid_im_R=np.real(hybrid_im_R)
hybrid_im_R=hybrid_im_R.astype('uint8')
finalimage=np.zeros([len(image2), len(image2[0]),3])
finalimage[:,:,0]=hybrid_im_B
finalimage[:,:,1]=hybrid_im_G
finalimage[:,:,2]=hybrid_im_R
cv2.imwrite("Q4_14_hybrid_near.jpg",finalimage)   

img = cv2.resize(finalimage, (130, 80), interpolation = cv2.INTER_AREA)
cv2.imwrite("Q4_15_hybrid_far.jpg",img)             