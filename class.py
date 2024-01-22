from os.path import splitext
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tensorflow.keras import models
from tensorflow.keras.layers import Layer, BatchNormalization
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import cv2
import easyocr

#Histogram eşitleme araştır.

class ComicTranslator:
    def __init__(self,first_img_path):
        
        self.first_img_path = first_img_path
        self.last_img_path = ""
        self.speech_only_img_path = ""
        self.edge_removed_net = ""
        self.edge_removed= ""
        
        
    def run(self):
        self.create_speech_bubbles()
        self.sub_two_image_to_get_speech_bubbles_only()
        #self.focus_text()
        self.detect_edges_or_build_blocks(1) # siyaha boyama
        self.remove_edges()
        #self.focus_text()
        self.detect_edges_or_build_blocks(2) # beyaza boyama
        self.detect_text_blocks()
        
        
        
        
    def create_speech_bubbles(self):
        class CustomBatchNormalization(Layer):
            def __init__(self, **kwargs):
                super(CustomBatchNormalization, self).__init__(**kwargs)
                self.bn_layer = BatchNormalization()

            def call(self, inputs, training=None):
                return self.bn_layer(inputs, training=training)

        # Define the custom_objects dictionary with CustomBatchNormalization
        custom_objects = {'CustomBatchNormalization': CustomBatchNormalization}


        img_path= self.first_img_path
        model_path = '0207_e500_std_model_4.h5'
        # Save path for prediction is based on the path of the image and will be
        # saved in the same directory
        prediction_save_path = splitext(img_path)[0] + '_pred.jpg'

        # Load the model with custom_objects
        model = models.load_model(model_path, custom_objects=custom_objects)

        img = imread(img_path)
        img = resize(img, (768, 512), anti_aliasing=True, preserve_range=True)
        img = np.expand_dims(img, axis=0)
        img = (img / 255.0).astype(np.float32)  # Convert to float32 before saving

        p = model.predict(img)

        # Address the low contrast warning by rescaling pixel values to [0, 255]
        p_image = (p[0, :, :, 0] * 255).astype(np.uint8)

        imsave(fname=prediction_save_path, arr=p_image)
        self.last_img_path = prediction_save_path

    def remove_edges(self):
        original_image =  self.resize2(cv2.imread(self.speech_only_img_path))
        filter_image =  self.resize2(cv2.imread(self.last_img_path))
        #deneme= cv2.imread("sub-result.png")

        # Ensure that both images have the same dimensions
        filter_image = cv2.resize(filter_image, (original_image.shape[1], original_image.shape[0]))
        #deneme = cv2.resize(deneme, (original_image.shape[1], original_image.shape[0]))

        # Convert images to the same data type
        original_image = original_image.astype(np.float32)
        filter_image = filter_image.astype(np.float32)

        
        # Subtract the filter_image from the original_image
        result_image = np.clip(original_image - filter_image, 0, 255).astype(np.uint8)
            
        

        # Display or save the result_image
        cv2.imshow("Result Image", result_image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        last_img_path = f"{self.first_img_path}-edge-removed.png"
        cv2.imwrite(last_img_path, result_image)
        
        
        self.last_img_path = last_img_path
        self.edge_removed = last_img_path
    
    def resize2(self,image):
        img_for_resize =  cv2.imread("./comic-pages/spidey.jpg")
        resized_img = cv2.resize(image, (img_for_resize.shape[1], img_for_resize.shape[0]))
        return resized_img
    
    def sub_two_image_to_get_speech_bubbles_only(self):
        original_image =  self.resize2(cv2.imread(self.first_img_path))
        filter_image = self.resize2(cv2.imread(self.last_img_path) ) 
        
       

        # Ensure that both images have the same dimensions
        filter_image = cv2.resize(filter_image, (original_image.shape[1], original_image.shape[0]))
        

        # Convert images to the same data type
        original_image = original_image.astype(np.float32)
        filter_image = filter_image.astype(np.float32)

        
        # Subtract the filter_image from the original_image
        result_image = np.clip(filter_image - original_image, 0, 255).astype(np.uint8)
            
        

        # Display or save the result_image
        cv2.imshow("Result Image", result_image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        last_img_path = f"{self.first_img_path}-speech-only.png"
        cv2.imwrite(last_img_path, result_image)
        
        
        self.last_img_path = last_img_path
        self.speech_only_img_path = last_img_path
    
    def detect_text_blocks(self):
        try:
            original_image = self.resize2(cv2.imread(self.edge_removed_net))
        except:
            original_image = self.resize2(cv2.imread(self.edge_removed))
            
        image = self.resize2(cv2.imread(self.last_img_path, cv2.IMREAD_GRAYSCALE))


        # Threshold uygula (beyazı belirginleştir)
        _, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

        vertical_blur = cv2.GaussianBlur(thresholded, (51, 51), 0)
        cv2.imshow('Result', vertical_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("blur.png",vertical_blur)
        
        cv2.imshow('Result', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Contours bul
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Her bir contour için döngü
        roi_sayac = 0
        for contour in contours:
            # Contour'un bir dikdörtgeni (bbox) içine alınması
            x, y, w, h = cv2.boundingRect(contour)

            # Eğer dikdörtgenin alanı belirli bir eşik değerinden büyükse
            if cv2.contourArea(contour) > 100:
                # Dikdörtgeni çiz
                roi = original_image[y:y + h, x:x + w]
                
                cv2.rectangle(original_image, (x, y), (x+w, y+h), (255, 255, 255), 2)
                img_text = pytesseract.image_to_string(roi,lang='eng')
                print(img_text)
                roi_sayac += 1
                print("----------------------")

        # Sonucu göster
        cv2.imshow('Result', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        last_img_path = f"{self.last_img_path}-text-blocks.png"
        cv2.imwrite(last_img_path, original_image)

            

    def detect_edges_or_build_blocks_new(self, metod):
        # Tesseract OCR dil modeli seçimi (örneğin 'eng' İngilizce demektir)
        language = 'eng'

        # Metni tespit etmek istediğiniz resmin dosya yolunu belirtin
        file_path = self.last_img_path  # Dosya yolunu kendi dosya yolunuzla değiştirin

        # Resmi oku
        image = self.resize2(cv2.imread(file_path))

        # Tesseract OCR ile metni tespit etme
        custom_config = r'--oem 3 --psm 6'  # Optional: Tesseract OCR ayarları
        results = pytesseract.image_to_boxes(image, lang=language, config=custom_config).splitlines()

        # Metni ve konumları tutmak için bir sözlük oluştur
        detected_text_dict = []

        for line in results:
            bbox = line.split()[1:5]
            text = line.split()[0]
            top_left = (int(bbox[0]), int(bbox[1]))
            bottom_right = (int(bbox[2]), int(bbox[3]))

            # Draw a white rectangle to fill the bounding box
            if metod == 1:
                image[top_left[1]-2:bottom_right[1]+2, top_left[0]-2:bottom_right[0]+2] = [0, 0, 0]  # siyah orginal
            else:
                image[top_left[1]-5:bottom_right[1]+5, top_left[0]-5:bottom_right[0]+5] = [255, 255, 255]  # beyaz orginal

        cv2.imshow('Image without text', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        last_img_path = f"{self.last_img_path}-edge.png"
        cv2.imwrite(last_img_path, image)

        self.last_img_path = last_img_path
    
    
    
    def detect_edges_or_build_blocks(self,metod):
        # easyocr ile dil modeli seçimi (örneğin 'en' İngilizce demektir)
        language = 'en'

        # easyocr tanıyıcıyı oluşturma
        reader = easyocr.Reader([language])

        # Metni tespit etmek istediğiniz resmin dosya yolunu belirtin
        file_path = self.last_img_path  # Dosya yolunu kendi dosya yolunuzla değiştirin

        # Resmi oku
        image = self.resize2(cv2.imread(file_path))

        # easyocr ile metni tespit etme
        results = reader.readtext(file_path)

        # Metni ve konumları tutmak için bir sözlük oluştur
        detected_text_dict = []

        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Draw a white rectangle to fill the bounding box
            if metod == 1:
                
                image[top_left[1]-2:bottom_right[1]+2, top_left[0]-2:bottom_right[0]+2] = [0, 0, 0] #siyah orginal
            else:
                image[top_left[1]-5:bottom_right[1]+5, top_left[0]-5:bottom_right[0]+5] = [255, 255, 255] #beyaz orginal
            


        cv2.imshow('Image without text', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        last_img_path = f"{self.last_img_path}-edge.png"
        cv2.imwrite(last_img_path, image)
        
        self.last_img_path = last_img_path
        
    def focus_text(self):
        # Resmi yükle
        image = self.resize2(cv2.imread(self.last_img_path))
        image = 255-image

        # Gri tonlamalı hale getir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Eşikleme uygula
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Görüntüyü göster

        cv2.imshow('Processed Image', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_text = pytesseract.image_to_string(255-thresh,lang='eng')
        print(img_text)
        last_img_path = f"{self.last_img_path}-net.png"
        cv2.imwrite(last_img_path, thresh)
        self.last_img_path = last_img_path
        self.edge_removed_net = last_img_path
            

        
translator = ComicTranslator(first_img_path='./comic-pages/complex.jpg')
translator.run()


