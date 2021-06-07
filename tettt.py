from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
import cv2
import os
import numpy

def dessiner(Image):
    label1 = Label(image=Image)
    #if not label1.image : None
    #label1.option_clear()
    label1.image = Image
    label1.pack(side=LEFT, padx=30, pady=30)



def InserImage(event=None):
    filepath = askopenfilename(title="Ouvrir une image", filetypes=[('png files', '.png'), ('all files', '.*')])
    #pohoto = PhotoImage(file=filepath)
    image = cv2.imread(filepath)
    b, g, r = cv2.split(image)
    img = cv2.merge((r, g, b))
    img = cv2.resize(img, (200, 200))  # Resize image
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    dessiner(imgtk)
    #dessiner(pohoto)
    im1 = Image.open(filepath)
    im1.save("pohoto.png")


    # Position image
    #canvas = Canvas(fen, width=photo.width(), height=photo.height())
    #canvas.create_image(0, 0, anchor=NW, image=photo)
    #canvas.pack()
#get the path of the directory
dir_path = os.path.dirname(os.path.realpath(__file__))
#create the Output folder if it does not exist
if not os.path.exists('Output'):
  os.makedirs('Output')
#import the models provided in the OpenCV repository
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
#loop through all the files in the folder




fen =Tk()
fen.geometry('800x500')



Frame2 = Frame(fen, borderwidth=1, background='yellow')
Frame2.pack(side=RIGHT, padx=30, pady=30)


bouton = Button(Frame2,text="fermer", command=fen.quit)

def UploadAction(event=None):
    for file in os.listdir(dir_path):
        # split the file name and the extension into two variales
        filename, file_extension = os.path.splitext(file)
        # check if the file extension is .png,.jpeg or .jpg to avoid reading other files in the directory
        if (file_extension in ['.png', '.jpg', '.jpeg']):
            # read the image using cv2
            image = cv2.imread(file)
            # accessing the image.shape tuple and taking the first two elements which are height and width
            (h, w) = image.shape[:2]
            # get our blob which is our input image after mean subtraction, normalizing, and channel swapping
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            # input the blob into the model and get back the detections from the page using model.forward()
            model.setInput(blob)
            detections = model.forward()
            # Iterate over all of the faces detected and extract their start and end points
            count = 0
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                confidence = detections[0, 0, i, 2]
                # if the algorithm is more than 16.5% confident that the detection is a face, show a rectangle around it
                if (confidence > 0.165):
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    count = count + 1
            # save the modified image to the Output folder
            cv2.imwrite('Output/' + file, image)
            b, g, r = cv2.split(image)
            img = cv2.merge((r, g, b))
            img = cv2.resize(img, (200, 200))  # Resize image
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im)
            dessiner(imgtk)
            # print out a success message
            Label(l, text="Face detection complete for image " + file + " (" + str(count) + ") faces found!").pack()
          #  print("Face detection complete for image " + file + " (" + str(count) + ") faces found!")



bouton2 = Button(Frame2,text="inserer image" , command=InserImage)
bouton2.pack(side=TOP, padx=5, pady=5)
bouton1 = Button(Frame2,text="traiter" , command=UploadAction)
bouton.pack(side=BOTTOM, padx=5, pady=5)
bouton1.pack(side=TOP, padx=5, pady=5)
l = LabelFrame(fen, text=" résultas ", padx=20, pady=20)
l.pack(side=BOTTOM, fill="both", expand="yes")

fen.mainloop()
