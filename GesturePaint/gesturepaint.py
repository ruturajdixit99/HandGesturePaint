import cv2
import numpy as np
import mediapipe as mp


class startPaint():

    def paint():
        # Initialize Mediapipe Hand model
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        # Initialize hand tracking
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

        # Initialize canvas
        canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255

        # Capture video feed
        cap = cv2.VideoCapture(0)

        # Previous finger position
        prev_x, prev_y = None, None

        # Color options
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),(0,128,128)]  # Green, Blue, Red
        current_color_index = 0
        current_color = colors[current_color_index]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and get hand landmarks
            result = hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Get coordinates of landmarks
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        lm_x = int(lm.x * frame_width)
                        lm_y = int(lm.y * frame_height)
                        lm_list.append((lm_x, lm_y))
                    
                    # Drawing mode: Index finger is up
                    if lm_list[8][1] < lm_list[6][1]:  # If tip of index finger is above the PIP joint
                        cur_x, cur_y = lm_list[8]
                        
                        if prev_x is None or prev_y is None:
                            prev_x, prev_y = cur_x, cur_y
                        
                        cv2.line(canvas, (prev_x, prev_y), (cur_x, cur_y), current_color, 5)
                        
                        prev_x, prev_y = cur_x, cur_y
                    else:
                        prev_x, prev_y = None, None
                    
                    # Drawing the landmarks and connections on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display the frame with hand landmarks
            win_name="HandTrackingFeed"
            cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name,720,1280)
            
            # cv2.imshow('Hand Tracking', frame)
            cv2.imshow(win_name,frame)
            
            # Display the canvas
            cv2.imshow('Hand Gesture Painter', canvas)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset the canvas
                canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
            elif key == ord('1'):
                current_color_index = 0
                current_color = colors[current_color_index]
            elif key == ord('2'):
                current_color_index = 1
                current_color = colors[current_color_index]
            elif key == ord('3'):
                current_color_index = 2
                current_color = colors[current_color_index]
            elif key == ord('4'):
                current_color_index = 3
                current_color = colors[current_color_index]

        cap.release()
        cv2.destroyAllWindows()
