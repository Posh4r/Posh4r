import mouse
import cv2
import math
import time
import numpy as np
from mss import mss
from PIL import Image
import win32api
import win32con
global mousedown
#SIMPLE RECOIL CONTROL FOR PUBG Mobile BASED ON FEATURE TRACKING ALGORITHMS
#xNWRx

#goodFeaturesToTrack
def getTrackingPoints(img, width=450, height=250, initialWidth=0, initialHeight=0):
    gray = cv2.cvtColor(img[initialHeight:initialHeight+height,initialWidth:initialWidth+width],cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,50,0.001,8)
    corners = np.int0(corners)
    return corners, gray

#DrawPoints - generic drawing function
def drawPoints(img, points, color):
    for i in points:
        x,y = i.ravel()
        #print(str(x) + " " + str(y) + " " + repr(img[200+y,450+x]))
        cv2.circle(img,(x+450,y+200),3,color,-1)        

#Euclidean distance, probably very naive
def euclideanDist(p1,p2):
    diffx = p1[0]-p2[0]
    diffy = p1[1]-p2[1]
    #print(str(diffx) + " " + str(diffy))
    return math.sqrt(diffx**2+diffy**2)

def fastPoints():
    fast = cv2.FastFeatureDetector_create()
    img = cv2.imread('test1.jpg')
    img_one = img
    kp, des = fast.detectAndCompute(img,None)
    #corners1, gray1 = getTrackingPoints(img,450,300,450,200)
    img = cv2.imread('test2.jpg')
    kp2, des = fast.detectAndCompute(img,None)
    #corners2, gray2 = getTrackingPoints(img,450,300,450,200)
    #drawPoints(img,kp,(0,255))
    #drawPoints(img,kp2,255)
    img = cv2.drawKeypoints(img,kp,None,color=(255,0,0))
    img = cv2.drawKeypoints(img,kp2,None,color=(0,0,255))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.imwrite('test1_goodFeatures.jpg',img)


def orbDetectionAndCompute(img, img2):
    #img = cv2.imread('test1.jpg')[200:500,450:900]
    #img2 = cv2.imread('test2.jpg')[200:500,450:900]
    orb = cv2.ORB_create()
    kp1 = orb.detect(img,None)
    kp2 = orb.detect(img2,None)

    kp1, des1 = orb.compute(img, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)#for orb we should use NORM_HAMMING as the matcher
    matches = bf.match(des1,des2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    #for m,n in matches:
     #   if m.distance < 0.75*n.distance:
      #      good.append([m])
       #     print()
    img2 = np.array(sct.grab(monitor).rgb)
    
    return matches, kp1, kp2


def leftClickDown():
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
def leftClickUp():
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
while 'playing':    
    with mss() as sct:
        #mhook = mouse.hook()
        #mouse.on_button(mousedown_func, args=(), buttons=('left'), types=('down'))
        #mouse.on_button(mouserelease_func, args=(), buttons=('left'), types=('up'))
        monitor = {'top': 200, 'left': 450, 'width': 450, 'height': 350}
        output = 'sct-{top}x{left}_{width}x{height}.png'.format(**monitor)
        old_img = cv2.UMat(np.array(sct.grab(monitor)))
        cv2.namedWindow("win1");
        cv2.moveWindow("win1", 1000,20);
        cv2.imshow('win1', old_img)
        # Grab the data
        #if mouse.is_pressed(button='left'):
                
        while mouse.is_pressed(button='middle'):
            last_time = time.time()
            #win32api.GetFocus()
            # Get raw pixels from the screen, save it to a Numpy array
            img = cv2.UMat(np.array(sct.grab(monitor)))
            matches, kp1, kp2 = orbDetectionAndCompute(old_img, img)
            
            # Display the picture
            #cv2.imshow('win1', result)
            #old_img = img
            xm = 0
            ym = 0
            for match in matches[:10]:
                x1,y1 = kp1[match.queryIdx].pt
                x2,y2 = kp2[match.trainIdx].pt
                xm += x2-x1
                ym += y2-y1
                
            xm = xm/10
            ym = ym/10
            print('trying to mousemove')
            
            leftClickDown()
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(xm*0.5), int(ym*0.6), 0, 0);
                
            time.sleep(0.01)
            leftClickUp()
            #old_img = img
            #mouse.press(button='left')
            #for i in range(10):
            #    mouse.move(xm/10, ym/10, absolute=False, duration=0.08)
            #mouse.drag(0, 0, xm, ym, absolute=False, duration=0.1)
            

            #mouse.release(button='left')
            
            print(str(xm) + " " + str(ym))
            
            # Display the picture in grayscale
            # cv2.imshow('OpenCV/Numpy grayscale',
            #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

            print('fps: {0}'.format(1 / (time.time()-last_time)))

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
