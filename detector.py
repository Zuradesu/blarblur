import cv2
import mediapipe as mp


class HandDetector:

    def __init__(self):

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.drawer = mp.solutions.drawing_utils

    def fingers_up(self, hand):

        fingers = []

        # Index
        fingers.append(hand.landmark[8].y < hand.landmark[6].y)

        # Middle
        fingers.append(hand.landmark[12].y < hand.landmark[10].y)

        # Ring
        fingers.append(hand.landmark[16].y < hand.landmark[14].y)

        # Pinky
        fingers.append(hand.landmark[20].y < hand.landmark[18].y)

        return fingers
    
    def is_peace(self, fingers):
        
        return fingers == [True, True, False, False]

    def detect(self, frame):

        peace = False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                self.drawer.draw_landmarks(
                    frame,
                    hand,
                    self.mpHands.HAND_CONNECTIONS
                )

                fingers = self.fingers_up(hand)

                if self.is_peace(fingers):
                    peace = True
                    print("✌️ Peace Detected!")

                print(fingers)

                for i, landmark in enumerate(hand.landmark):

                    h, w, _ = frame.shape

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.putText(
                        frame,
                        str(i),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        return frame, peace