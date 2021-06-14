from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
import cv2
import os
import numpy as np


#get the path of the directory
dir_path = os.path.dirname(os.path.realpath(__file__))
#create the Output folder if it does not exist
if not os.path.exists('Output'):
  os.makedirs('Output')
#import the models provided in the OpenCV repository
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')

# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')

def dessiner(Image):
    label1 = Label(image=Image)
    label1.image = Image
    label1.pack(side=LEFT, padx=30, pady=30)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade, eyeCascade, noseCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords = draw_boundary(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
        coords = draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
    return img

def PasserLive(Event=None):



    video_capture = cv2.VideoCapture(0)

    while True:
        # Reading image from video stream
        _, img = video_capture.read()
        # Call method we defined above
        img = detect(img, faceCascade, eyesCascade, noseCascade)
        # Writing processed image in a new window
        cv2.imshow("face detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # releasing web-cam
    video_capture.release()
    # Destroying output window
    cv2.destroyAllWindows()


def InserImage(event=None):
    filepath = askopenfilename(title="Ouvrir une image", filetypes=[('png files', '.png'), ('all files', '.*')])
    image = cv2.imread(filepath)
    b, g, r = cv2.split(image)
    img = cv2.merge((r, g, b))
    img = cv2.resize(img, (300, 300))  
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    dessiner(imgtk)
    im1 = Image.open(filepath)
    im1.save("pohoto.png")






fen =Tk()
fen.title('projet de fin d étude detection de visage, yeux et le nez')
fen.geometry('1000x500')



Frame2 = Frame(fen, borderwidth=1, background='yellow')
Frame2.pack(side=RIGHT, padx=30, pady=30)




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
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
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
            img = cv2.resize(img, (300, 300))  # Resize image
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im)
            dessiner(imgtk)
            # print out a success message
            Label(l, text="Face detection complete for image " + file + " (" + str(count) + ") faces found!").pack()




bouton4 = Button(Frame2, text="      Quitter      ", command=fen.quit, relief=GROOVE )
bouton3 = Button(Frame2,text=" passer en live " , command=PasserLive, relief=SUNKEN)
bouton3.pack(side=TOP, padx=5, pady=5)
bouton2 = Button(Frame2,text=" insérer image " , command=InserImage, relief=GROOVE)
bouton2.pack(side=TOP, padx=5, pady=5)
bouton1 = Button(Frame2,text="       traiter      ", command=UploadAction, relief=SUNKEN)
bouton4.pack(side=BOTTOM, padx=5, pady=5)
bouton1.pack(side=TOP, padx=5, pady=5)
l = LabelFrame(fen, text=" résultas ", padx=20, pady=20)
l.pack(side=BOTTOM, fill="both", expand="yes")






fen.mainloop()
