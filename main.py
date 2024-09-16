import os
import cv2 as cv
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

c_fucsia = (255, 0, 255)
c_green = (0, 255, 0)
model_path = r'..\model\lego.keras'
predict_path = 'Data\\predict'

def plot_images(images, predicted_labels, num_images, class_names):
    plt.figure(figsize=(24, 24))
    for i in range(num_images):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i])
        predicted_label = class_names[predicted_labels[i]]
        plt.title(f'Pred: {predicted_label}', fontsize=20)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filename = r'Data\img\Lego_2.JPG'
    if not os.path.exists(filename):
        print('Lego messi nella cartella sbagliata')
        exit(1)
        
    print('Carico Lego')
    
    img = cv.imread(filename)
    img_resize = cv.resize(img , (850,850))
    
    gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
    
    cv.imshow('Originale', img_resize)
    cv.imshow('Scala di grigi', gray)
    
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
    
    final = cv.medianBlur(adaptive, 7)
    cv.imshow('Adaptive Threshold', adaptive)
    cv.imshow('Adaptive Threshold Blurred', final)
    
    blur = cv.GaussianBlur(gray, (7, 7), 0)
    _, otsu = cv.threshold(blur, 0, 255, flag + cv.THRESH_OTSU)
    cv.imshow('Otsu\'s Threshold Blurred', otsu)

    contours, hierarchy = cv.findContours(otsu,
                                          mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE,
                                          )
    
    
    check = img_resize.copy()
    c_black = (0, 0, 0)
    check[::] = c_black
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
        
        target = img_resize[y-20:y+h+20,x-20:x+w+20]
        cv.drawContours(check , [cnt], -1, (0, 0, 255), 3)
        
        cv.imwrite(f"{predict_path}\\target_{i}.png", target)
        cv.imshow(f'target: {i}', target)
        
        cv.drawContours(check, [cnt], 0, c_green, 3)
        valid_contours.append(cnt)
        
        
    cv.imshow('Check', check)
    
    img3 = cv.drawContours(img_resize, valid_contours, -1, c_green, 3)
    cv.imshow('Contorni', img3)

    print(f'Lego trovati: {len(valid_contours)}')

    
    df = pd.read_csv(r'./validation.csv', header=0)
    df_selected = df.groupby('Id').first()

    labels = df_selected['Description'].values
    print(labels)

    model = load_model(model_path)
    
    img_path = []
    for img in os.listdir(predict_path):
        img_path = os.path.join(img, predict_path)
    
    y_hat_pre = model.predict(img_path)

    predicted_labels = y_hat_pre.argmax(axis=1)
        
    num_images = min(20, len(img_path))
    plot_images(img_path, predicted_labels, num_images, labels)
    
    cv.waitKey(0)
    cv.destroyAllWindows()