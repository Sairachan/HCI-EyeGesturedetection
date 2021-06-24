import cv2
import dlib
import numpy as np
from math import hypot

cap = cv2.VideoCapture(0)
board = np.zeros((300, 1400), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

keyboard = np.zeros((600, 1000, 3), np.uint8)
keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "<"}
keys_set_2 = {0: "Y", 1: "U", 2: "I", 3: "O", 4: "P",
              5: "H", 6: "J", 7: "K", 8: "L", 9: "_",
              10: "V", 11: "B", 12: "N", 13: "M", 14: "<"}

def letter(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400

    width = 200
    height = 200
    th = 3  # thickness
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4 # thickness lines
    cv2.line(keyboard, (int(cols/2) - int(th_lines/2), 0),(int(cols/2) - int(th_lines/2), rows),
             (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 300), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (80 + int(cols/2), 300), font, 6, (255, 255, 255), 5)

def mid(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_PLAIN

imgs = 0
letter_index = 0
blinking_imgs = 0
imgs_to_blink = 6
imgs_active_letter = 9

# Text and keyboard settings
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_imgs = 0

while True:
    _, img = cap.read()
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    rows, cols, _ = img.shape
    keyboard[:] = (0, 0, 0)
    imgs += 1
    new_img = np.zeros((500, 500, 3), np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img[rows - 50: rows, 0: cols] = (255, 255, 255)

    if select_keyboard_menu is True:
        draw_menu()

    # Keyboard selected
    if keyboard_selected == "left":
        keys_set = keys_set_1
    else:
        keys_set = keys_set_2
    active_letter = keys_set[letter_index]

    faces = detector(gray)
    for face in faces:
        print(face)
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)

        left_point1 = (landmarks.part(36).x, landmarks.part(36).y)
        right_point1 = (landmarks.part(39).x, landmarks.part(39).y)
        centreup1 = mid(landmarks.part(37), landmarks.part(38))
        centredown1 = mid(landmarks.part(41), landmarks.part(40))

        left_point2 = (landmarks.part(42).x, landmarks.part(42).y)
        right_point2 = (landmarks.part(45).x, landmarks.part(45).y)
        centreup2 = mid(landmarks.part(43), landmarks.part(44))
        centredown2 = mid(landmarks.part(47), landmarks.part(46))

        verline_length1 = hypot((centreup1[0] - centredown1[0]), (centreup1[1] - centredown1[1]))+ 1
        horline_length1 = hypot((left_point1[0] - right_point1[0]), (left_point1[1] - right_point1[1]))

        ###############################
        # Blinking detection
        ratio1 = horline_length1 / verline_length1

        verline_length2 = hypot((centreup2[0] - centredown2[0]), (centreup2[1] - centredown2[1])) + 1
        horline_length2 = hypot((left_point2[0] - right_point2[0]), (left_point2[1] - right_point2[1]))

        ratio2 = horline_length2 / verline_length2

        if ratio1 > 4.5 and ratio2 > 4.5:
            cv2.putText(img, "Blink", (50, 150), font, 5, (255, 0, 0))

        ################################
        # Gaze detection left eye
        left_eyex = np.array([(landmarks.part(36).x), (landmarks.part(37).x),
                              (landmarks.part(38).x), (landmarks.part(39).x),
                              (landmarks.part(40).x), (landmarks.part(41).x)])

        left_eyey = np.array([(landmarks.part(36).y), (landmarks.part(37).y),
                              (landmarks.part(38).y), (landmarks.part(39).y),
                              (landmarks.part(40).y), (landmarks.part(41).y)])

        minx1 = np.min(left_eyex)
        maxx1 = np.max(left_eyex)
        miny1 = np.min(left_eyey)
        maxy1 = np.max(left_eyey)

        eye1 = img[miny1:maxy1, minx1:maxx1]
        grayeye1 = cv2.cvtColor(eye1, cv2.COLOR_BGR2GRAY)
        _, bweye1 = cv2.threshold(grayeye1, 45, 255, cv2.THRESH_BINARY)

        height1, width1 = bweye1.shape

        leftbw1 = bweye1[0:height1, 0:int(width1 / 2)]
        leftwhite1 = cv2.countNonZero(leftbw1)

        rightbw1 = bweye1[0:height1, int(width1 / 2):width1]
        rightwhite1 = cv2.countNonZero(rightbw1)

        if leftwhite1 == 0:
            gazeratio1 = 1
        elif rightwhite1 == 0:
            gazeratio1 = 5
        else:
            gazeratio1 = leftwhite1 / rightwhite1

        eye1 = cv2.resize(eye1, None, fx=5, fy=5)
        bweye1 = cv2.resize(bweye1, None, fx=5, fy=5)
        leftbw1 = cv2.resize(leftbw1, None, fx=5, fy=5)
        rightbw1 = cv2.resize(rightbw1, None, fx=5, fy=5)

        cv2.imshow("Left Eye", eye1)
        cv2.imshow("Left B/W eye", bweye1)
        cv2.imshow("Left Left B/W", leftbw1)
        cv2.imshow("Left Right B/W", rightbw1)
        ##############################################
        # Gaze detection right
        right_eyex = np.array([(landmarks.part(42).x), (landmarks.part(43).x),
                               (landmarks.part(44).x), (landmarks.part(45).x),
                               (landmarks.part(46).x), (landmarks.part(47).x)])

        right_eyey = np.array([(landmarks.part(42).y), (landmarks.part(43).y),
                               (landmarks.part(44).y), (landmarks.part(45).y),
                               (landmarks.part(46).y), (landmarks.part(47).y)])

        minx2 = np.min(right_eyex)
        maxx2 = np.max(right_eyex)
        miny2 = np.min(right_eyey)
        maxy2 = np.max(right_eyey)

        eye2 = img[miny2:maxy2, minx2:maxx2]
        grayeye2 = cv2.cvtColor(eye2, cv2.COLOR_BGR2GRAY)
        _, bweye2 = cv2.threshold(grayeye2, 45, 255, cv2.THRESH_BINARY)

        height2, width2 = bweye2.shape

        leftbw2 = bweye2[0:height2, 0:int(width2 / 2)]
        leftwhite2 = cv2.countNonZero(leftbw2)

        rightbw2 = bweye2[0:height2, int(width2 / 2):width2]
        rightwhite2 = cv2.countNonZero(rightbw2)

        if leftwhite2 == 0:
            gazeratio2 = 1
        elif rightwhite2 == 0:
            gazeratio2 = 5
        else:
            gazeratio2 = leftwhite2 / rightwhite2

        eye2 = cv2.resize(eye2, None, fx=5, fy=5)
        bweye2 = cv2.resize(bweye2, None, fx=5, fy=5)
        leftbw2 = cv2.resize(leftbw2, None, fx=5, fy=5)
        rightbw2 = cv2.resize(rightbw2, None, fx=5, fy=5)

        cv2.imshow("Right Eye", eye2)
        cv2.imshow("Right B/W eye", bweye2)
        cv2.imshow("Right Left B/W", leftbw2)
        cv2.imshow("Right Right B/W", rightbw2)

        ###############################
        gazeaverage = (gazeratio2 + gazeratio1) / 2

        if gazeaverage <= 1:
            cv2.putText(img, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_img[:] = (0, 0, 255)
        elif gazeaverage < 1.7 and gazeaverage > 1:
            cv2.putText(img, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            new_img[:] = (255, 0, 0)
            cv2.putText(img, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)

        if select_keyboard_menu is True:
        # Detecting gaze to select Left or Right keybaord
            gaze_ratio = gazeaverage

            if gaze_ratio <= 0.9:
                keyboard_selected = "right"
                keyboard_selection_imgs += 1

                if keyboard_selection_imgs == 15:
                    select_keyboard_menu = False

                    # Set frames count to 0 when keyboard selected
                    imgs = 0
                    keyboard_selection_imgs = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_imgs = 0
            else:
                keyboard_selected = "left"
                keyboard_selection_imgs += 1
                # If Kept gaze on one side more than 15 frames, move to keyboard
                if keyboard_selection_imgs == 15:
                    select_keyboard_menu = False
                    # Set frames count to 0 when keyboard selected
                    imgs = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_imgs = 0

        else:
            # Detect the blinking to select the key that is lighting up
            blinking_ratio= (ratio1+ratio2)/2
            if blinking_ratio > 4.5:
            # cv2.putText(img, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
                blinking_imgs += 1
                imgs -= 1


                # Typing letter
                if blinking_imgs == imgs_to_blink:
                    if active_letter != "<" and active_letter != "_":
                        text += active_letter
                    if active_letter == "_":
                        text += " "
                    select_keyboard_menu = True

            else:
                blinking_imgs = 0


        # Display letters on the keyboard
        if select_keyboard_menu is False:
            if imgs == imgs_active_letter:
                letter_index += 1
                imgs = 0
            if letter_index == 15:
                letter_index = 0
            for i in range(15):
                if i == letter_index:
                    light = True
                else:
                    light = False
                letter(i, keys_set[i], light)
        # Show the text we're writing on the board
        cv2.putText(board, text, (80, 100), font, 9, 0, 3)

        # Blinking loading bar
        percentage_blinking = blinking_imgs / imgs_to_blink
        loading_x = int(cols * percentage_blinking)
        cv2.rectangle(img, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)



    cv2.imshow("New Image", new_img)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Video", img)
    cv2.imshow("Board", board)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

