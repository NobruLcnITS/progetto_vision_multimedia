import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

c_fucsia = (255, 0, 255)
c_green = (0, 255, 0)
model_path = r'..\model\lego_1_04.keras'
predict_path = 'Data\\predict'


def process(img: np.ndarray) -> list:
    
    if img.shape > (400, 800, 3):
        img_resize = cv.resize(img , (850,850))
    else :
        img_resize = img.copy()
        
    img_resize_RGB = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img_resize_RGB, cv.COLOR_BGR2GRAY)
    
    print(f'Media: {np.mean(gray)}')
    avg = np.mean(gray)
    
    if avg > 125:
        flag = cv.THRESH_BINARY_INV
        
    else:
        flag = cv.THRESH_BINARY
        
    
    adaptive = cv.adaptiveThreshold(gray, 255,
                                    cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY_INV,
                                    blockSize=11,
                                    C=6)
    
    blur = cv.GaussianBlur(gray, (7, 7), 0)
    _, otsu = cv.threshold(blur, 0, 255, flag + cv.THRESH_OTSU)
   

    contours, hierarchy = cv.findContours(otsu,
                                          mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE,
                                          )
    
    
    check = img_resize_RGB.copy()
    c_black = (0, 0, 0)
    check[::] = c_black 
    
    targets = []
    valid_contours = []
    
    for i, cnt in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(cnt)
        if w < 30 or h < 30:
            continue
        text_x = x
        text_y = y
        cv.putText(check,
                   text=f'{i+1}',
                   org=(text_x, text_y),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1,
                   color=c_fucsia,  
                   thickness=3) 
        x1 = max(0, x - 20)
        y1 = max(0, y - 20)
        x2 = min(img_resize_RGB.shape[1], x + w + 20)
        y2 = min(img_resize_RGB.shape[0], y + h + 20)
        target = img_resize_RGB[y1: y2, x1:x2]
        cv.drawContours(check , [cnt], -1, (0, 255, 0), 3)
        valid_contours.append(cnt)
        
        
        #cv.imwrite(f"{predict_path}\\target_{i}.png", target)

        target_RGB= cv.cvtColor(target, cv.COLOR_BGR2RGB)
        
        targets.append(target_RGB)
    
    test = np.median(targets[1], axis=-1)
        
    contorni = cv.drawContours(img_resize_RGB, valid_contours, -1, c_green, 3)
    contorni = cv.cvtColor(contorni, cv.COLOR_BGR2RGB)    
     
    return [img_resize,gray,adaptive,otsu,check,contorni]+targets, test


if __name__ == '__main__':
    demo = gr.Interface(
    fn=process,
    inputs=['image'],
    outputs=[gr.Gallery(),'text'],
    )
        
    demo.launch(share=True)