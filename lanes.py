import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] 
    y2 = int(y1*(3/5))#resultierenden linien enden alle auf der gleichen hoehe
    x1 = int((y1 - intercept) / slope)#x-werte werden mit der geradengleichung berechnet
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])
    

def average_slope_intercept(image,lines):
    left_fit = [] #erstellen von zwei listen
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4) #Linien werden in koordinaten dargestellt
        parameters = np.polyfit((x1, x2), (y1, y2), 1) #es werden jeweils die steigung und der 
        #y-achsenabschnitt berechnet
        slope = parameters[0] #steigung wird zugeordnet
        intercept = parameters[1] #y-achsenabschnitt wird zugeordnet
        if slope > 0: 
            left_fit.append((slope, intercept)) #wenn die steigung positiv ist, ist die linie rechts
        else:
            right_fit.append((slope, intercept)) #wenn die steigung negativ ist, ist die linie links
    left_fit_average = np.average(left_fit, axis=0) #Es wird eine Geradengleichung aufgestellt mit den Durchschnittswerten
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)#Die Durchschnittsglaichungen werden in zwei koordinaten angegeben
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])#Die beiden werte werden ausgegeben
    


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Es wird eine Schwarz-Weißes Bild erstellt
    canny = cv2.Canny(gray, 50, 150) #Die Kontraste werden bestimmt
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image) #Ein neues Bild fuer die Lienien wird erstellt
    if lines is not None:
        for line in lines: 
            x1, y1, x2, y2 = line.reshape(4)#Jede Linie die erkannt wird mit in zwei Koordinaten definiert
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) #Die definierte Linie wird eingezeichnet
    return line_image

def region_of_interest(image):
    height = image.shape[0] #hoehe des Bildes Wird bestimmt
    polygons = np.array([[(200, height), (1100, height),(550, 250)]]) #Die Koordinaten des Dreiecks 
    #werden in einem Array gespeichert
    mask = np.zeros_like(image) #erzeugt ein Schwarzes Bild mit der gleichen aufloesung wie die Anderen Bilder
    cv2.fillPoly(mask, polygons, 255) #Im schwarzen Bild wird das vorher im Array definierte dreieck eingesetzt
    masked_image = cv2.bitwise_and(image, mask) # Beide Bilder werden uebereinander gelegt
    return masked_image #nur der region  of interest wird ausgegeben

image = cv2.imread('test_image2.jpg') #Bild wird eingelesen
lane_image = np.copy(image) #Es wird eine Kopie erstellt
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), 40, 5) #Die schnittpunkte im 
#Hough Diagramm werden gezaehlt und die Geraden Bestimmt.
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1.5, 1)
cv2.imshow("result", combo_image)
cv2.waitKey(0)
