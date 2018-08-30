import cv2

# def convertToRGB(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def findFaces(colored_img, scalefactor = 1.1, neighbors = 5):
    # colored_img = cv2.imread('data/test1.jpg')
  #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()          
  
    #get the training set
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=neighbors);          
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (11, 188, 220), 2)              
    
    print ('number of faces:', len(faces))
    
    return img_copy
    
def findEyes(colored_img, scalefactor = 1.1, neighbors = 5):
    # colored_img = cv2.imread('data/test1.jpg')
  #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()          
  
    #get the training set
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
    
    #let's detect multiscale (some images may be closer to camera than others) images
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=neighbors);          
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in eyes:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)              
    
    print ('number of eyes:', len(eyes))
    
    return img_copy
    
def findSmiles(colored_img, scalefactor = 1.175, neighbors = 26):
    # colored_img = cv2.imread('data/test1.jpg')
  #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()          
  
    #get the training set
    smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')
    
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
    
    #let's detect multiscale (some images may be closer to camera than others) images
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=neighbors);          
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in smiles:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)              
    
    print ('number of eyes:', len(smiles))
    
    return img_copy