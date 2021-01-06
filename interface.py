import tkinter as tk
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Interface(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Reconnaissance de chiffre')
        self.geometry('350x450')
        self.configure(bg='#EDD4EF')

        self.ecrire()
        self.commandes()


    def ecrire(self):
        frame_dessin = tk.Frame(self)
        frame_dessin.pack(pady=20)
        self.canvas = tk.Canvas(frame_dessin, width=300, height=300, bg='white')
        self.canvas.pack()
        self.canvas.bind( "<B1-Motion>", self.paint )


    def paint(self, event):
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        self.canvas.create_oval( x1, y1, x2, y2, width=10) 


    def commandes(self):
        frame_bouton = tk.Frame(self, bg='#EDD4EF')
        frame_bouton.pack()
        bouton_prediction = tk.Button(frame_bouton, text='Prédire', command=self.prediction)
        bouton_prediction.grid(row=0, column=0, padx=20)
        bouton_clean = tk.Button(frame_bouton, text='Nettoyer', command=self.clean)
        bouton_clean.grid(row=0, column=1, padx=20)
        self.label_prediction = tk.Label(frame_bouton, text='', bg='#EDD4EF')
        self.label_prediction.grid(row=1, columnspan=2)

    def prediction(self):
        self.canvas.postscript(file = f'number.eps')
        img = Image.open(f'number.eps')
        img.save(f'number' + '.png', 'png') 
        image = cv2.imread(f'number.png', 0)
        image = cv2.resize(image, (28, 28))
        # cv2.imshow('Image', image) 
        # cv2.waitKey(0)
        image = image.reshape(-1, 28, 28, 1)
        model = tf.keras.models.load_model('reco_chiffre')
        prediction = model.predict(image)
        prediction = np.argmax(prediction, axis=1)[0]
        self.label_prediction.configure(text=prediction, font=('Helvetica', 20))


    def clean(self):
        self.canvas.delete("all")
        self.label_prediction.configure(text='')


app = Interface()
app.mainloop()