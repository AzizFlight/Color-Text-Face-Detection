

# import libraries
from imutils.video import VideoStream
from gtts import gTTS 
import cv2
import pytesseract
import imutils
import time
import os
import numpy as np
from pygame import mixer
import pygame

key = input() # To sore the user's input

# Function to remove items from a list
def removeFromList(mylist,removes):
    for item in removes:
        try:
            mylist.remove(item)
        except:
            pass
    try:
        mylist = list(filter(lambda v: v is not "", mylist))
    except:
        pass
    return mylist

# Function to convert a list to a string
def listToString(s):
    # Initialize an empty string
    str1 = ""

    # Add the elements in the list to the string
    for ele in s:
        str1 += ele

    # return string
    return str1



face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/haarcascade_frontalface_default.xml') # xml file tained for frontal face detection


# Start the camera
print("[INFO] waiting for camera to warmup...")
vs = VideoStream(-1).start()
time.sleep(2.0)

# Range for red 
red_lower = np.array([136, 87, 111], np.uint8) 
red_upper = np.array([180, 255, 255], np.uint8)

# Range for yellow 
yellow_lower = np.array([20, 100, 100], np.uint8) 
yellow_upper = np.array([30, 255, 255], np.uint8) 

# Range for green
green_lower = np.array([36, 25, 25], np.uint8)
green_upper= np.array([70,255,255], np.uint8)

# Range for blue
blue_lower = np.array([94,80,2], np.uint8) 
blue_upper = np.array([120, 255, 255], np.uint8) 


# Start the program
while True:
    
    # Read video from camera in image frames
    frame = vs.read()
    
    # --------------------------------Color Detection--------------------------------
    # When user presses c
    if key==ord("c"):
        
        # Convert image frame from RGB to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for each color
        red_mask = cv2.inRange(hsv, red_lower, red_upper)  
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) 
        green_mask= cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # A 5*5 kernel matrix of ones
        kernel = np.ones((5, 5), "uint8") 

        # Remove noise from the image for each color
        red_mask = cv2.dilate(red_mask, kernel) 
        res_red= cv2.bitwise_and(frame,frame,mask=red_mask)
        yellow_mask = cv2.dilate(yellow_mask, kernel)
        res_yellow = cv2.bitwise_and(frame, frame, mask = yellow_mask) 
        green_mask = cv2.dilate(green_mask, kernel) 
        res_green = cv2.bitwise_and(frame, frame, mask = green_mask) 
        blue_mask = cv2.dilate(blue_mask, kernel) 
        res_blue = cv2.bitwise_and(frame, frame, mask = blue_mask)
        
	# Create a contour to track red object
        contours,hierarchy=cv2.findContours(red_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                # Draw a rectangle around the red object with the corresponding dimensions
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(frame, (x, y), 
									(x + w, y + h), 
									(0, 0, 255), 2) 

                # Display descritpive text on the object
                cv2.putText(imageFrame, "Red Colour", (x, y), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
						(0, 0, 255))	

                # Convert the text to an mp3 audio file using Google library(internet connection is needed)
                language = 'en'
                mytext='Red object'
                myobj = gTTS(text=mytext, lang=language, slow=False)
                myobj.save("output.mp3")
                
                # Play the converted file
                pygame.mixer.init()
                pygame.mixer.music.load("output.mp3")
                pygame.mixer.music.play()
        
        # Create a contour to track yellow object
        contours, hierarchy = cv2.findContours(yellow_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                # Draw a rectangle around the red object with the corresponding dimensions
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(frame, (x, y), 
									(x + w, y + h), 
									(255, 0, 0), 2)

                # Display descriptive text on the object
                cv2.putText(imageFrame, "yellow Colour", (x, y), 
						cv2.FONT_HERSHEY_SIMPLEX, 
						1.0, (255, 0, 0))

                # Convert the text to an mp3 audio file using Google library(internet connection is needed)
                language = 'en'
                mytext='Yellow object' 
                myobj = gTTS(text=mytext, lang=language, slow=False)
                myobj.save("output.mp3")
                
                # Play the converted file
                pygame.mixer.init()
                pygame.mixer.music.load("output.mp3")
                pygame.mixer.music.play()
                 
        # Create a contour to track green object
        contours, hierarchy = cv2.findContours(green_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                # Draw a rectangle around the red object with the corresponding dimensions
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(frame, (x, y), 
									(x + w, y + h), 
									(255, 0, 0), 2)

                # Display descriptive text on the object
                cv2.putText(imageFrame, "green Colour", (x, y), 
						cv2.FONT_HERSHEY_SIMPLEX, 
						1.0, (255, 0, 0))

                # Convert the text to an mp3 audio file using Google library(internet connection is needed)
                language = 'en'
                mytext='Green object' 
                myobj = gTTS(text=mytext, lang=language, slow=False)
                myobj.save("output.mp3")
                
                # Play the converted file
                pygame.mixer.init()
                pygame.mixer.music.load("output.mp3")
                pygame.mixer.music.play()
                contours, hierarchy = cv2.findContours(blue_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 

        # Create a contour to track blue object
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                # Draw a rectangle around the red object with the corresponding dimensions
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(frame, (x, y), 
							(x + w, y + h), 
							(255, 0, 0), 2)

                # Display descriptive text on the object
                cv2.putText(imageFrame, "Blue Colour", (x, y), 
				            cv2.FONT_HERSHEY_SIMPLEX, 
				            1.0, (255, 0, 0))

                # Convert the text to an mp3 audio file using Google library(internet connection is needed)
                language = 'en'
                mytext='Blue object'
                myobj = gTTS(text=mytext, lang=language, slow=False)
                myobj.save("output.mp3")
                # Play the converted file
                pygame.mixer.init()
                pygame.mixer.music.load("output.mp3")
                pygame.mixer.music.play()
	
    # --------------------------------Text Detection--------------------------------
    # When user presses t
    elif key == ord("t"):
        # configuration
        config = ('-l eng --oem 1 --psm 3')

        # read text from image and store it as a string
        text = pytesseract.image_to_string(frame ,lang='eng',config=config)

        # Clean text from punctuation and noise
        mytext = text.split('\n')
        mytext = removeFromList(mytext, ['', ' ', '  ', '\"', '\x0c'])
        mytext = listToString(mytext)
        print(mytext)
        
        # If text is detected
        if len(mytext)!=0 :
            # Convert the text to audio using the Google library(intenet connection is needed)
            language = 'en'
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save("output.mp3")

            # Play the converted file
            pygame.mixer.init()
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()

        # If there is no text
        else :
            # Display and read error message
            print("no text")
            language = 'en'
            mytext='Please put your text in front of the camera'
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save("output.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()
            
    # --------------------------------Frontal Face Detection--------------------------------        
    elif key == ord("f"):
        # Convert captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the frontal faces using the xml cascade we created above
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
        # Draw the rectangle around each face with the corresponding dimensions
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
           
            # Inform user that there is a human using Google library(internt connection is needed)
            language = 'en'
            mytext='This is a human'
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save("output.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()
       




    # Program termination
    cv2.imshow("Frame", frame)

    # if [ESC] key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break


# Destroy all windows and quit program
print("\n [INFO] Exiting Program and cleanup stuff \n")
cv2.destroyAllWindows()
vs.stop()
