# globalPlugins/faceGuide.py

import sys
import os
import time
import threading

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "lib"
    )
)

import cv2

import globalPluginHandler
import scriptHandler
import ui
import config

from tones import beep


class GlobalPlugin(globalPluginHandler.GlobalPlugin):

    def __init__(self):
        super().__init__()

        config.conf.spec["faceGuide"] = {

            "guidanceStyle": "string(default='direction')",

            "feedbackMode": "string(default='speechAndTone')",

            "verbosity": "string(default='normal')",

            "lightingFeedback": "boolean(default=True)",

            "distanceFeedback": "boolean(default=True)"
        }

        self.settings = config.conf["faceGuide"]

        self.running = False
        self.selfieMode = False

        self.cap = None
        self.thread = None

        self.captureInProgress = False

        self.lastSpeech = ""

        self.lastFeedbackTimes = {}

        self.centerStartTime = None
        self.centerConfirmed = False

    # =====================================================
    # SAVE SETTINGS
    # =====================================================

    def saveSettings(self):

        config.conf["faceGuide"] = self.settings

        config.conf.save()

    # =====================================================
    # TERMINATE
    # =====================================================

    def terminate(self):

        self.stopCamera()

        super().terminate()

    # =====================================================
    # START CAMERA
    # =====================================================

    def startCamera(self):

        if self.running:

            ui.message(
                "Face guide already running"
            )

            return

        self.running = True

        self.thread = threading.Thread(
            target=self.cameraLoop,
            daemon=True
        )

        self.thread.start()

        ui.message(
            "Face guide started"
        )

    # =====================================================
    # STOP CAMERA
    # =====================================================

    def stopCamera(self):

        self.running = False

        self.captureInProgress = False

        if self.cap:

            self.cap.release()
            self.cap = None

        ui.message(
            "Face guide stopped"
        )

    # =====================================================
    # PLAY TONES
    # =====================================================

    def playTone(
        self,
        direction,
        intensity=1
    ):

        try:

            if direction == "left":

                beep(
                    350,
                    max(40, 220 - intensity * 30)
                )

            elif direction == "right":

                beep(
                    850,
                    max(40, 220 - intensity * 30)
                )

            elif direction == "up":

                beep(
                    1200,
                    100
                )

            elif direction == "down":

                beep(
                    250,
                    120
                )

            elif direction == "center":

                beep(
                    1000,
                    180
                )

        except:
            pass

    # =====================================================
    # FEEDBACK
    # =====================================================

    def sendFeedback(
        self,
        category,
        value=None,
        intensity=1
    ):

        cooldown = 1.2

        currentTime = time.time()

        if category in self.lastFeedbackTimes:

            elapsed = (
                currentTime
                - self.lastFeedbackTimes[
                    category
                ]
            )

            if elapsed < cooldown:

                return

        self.lastFeedbackTimes[
            category
        ] = currentTime

        spokenText = ""

        if category == "direction":

            style = self.settings[
                "guidanceStyle"
            ]

            if style == "direction":

                spokenMap = {

                    "left":
                    "Move left",

                    "right":
                    "Move right",

                    "up":
                    "Move up",

                    "down":
                    "Move down"
                }

            else:

                spokenMap = {

                    "left":
                    "9 o'clock",

                    "right":
                    "3 o'clock",

                    "up":
                    "12 o'clock",

                    "down":
                    "6 o'clock"
                }

            spokenText = spokenMap.get(
                value,
                ""
            )

        elif category == "centered":

            spokenText = (
                "Face centered"
            )

        elif category == "distance":

            spokenText = value

        elif category == "lighting":

            spokenText = value

        elif category == "capture":

            spokenText = value

        elif category == "noface":

            spokenText = (
                "No face detected"
            )

        mode = self.settings[
            "feedbackMode"
        ]

        if mode == "speech":

            if spokenText:

                ui.message(
                    spokenText
                )

        elif mode == "tone":

            if category == "direction":

                self.playTone(
                    value,
                    intensity
                )

            elif category == "centered":

                self.playTone(
                    "center"
                )

        elif mode == "speechAndTone":

            if (
                spokenText
                and spokenText
                != self.lastSpeech
            ):

                ui.message(
                    spokenText
                )

                self.lastSpeech = (
                    spokenText
                )

            if category == "direction":

                self.playTone(
                    value,
                    intensity
                )

            elif category == "centered":

                self.playTone(
                    "center"
                )

    # =====================================================
    # DIRECTION
    # =====================================================

    def getDirectionFeedback(
        self,
        faceX,
        faceY,
        frameWidth,
        frameHeight
    ):

        centerX = frameWidth / 2
        centerY = frameHeight / 2

        toleranceX = (
            frameWidth * 0.10
        )

        toleranceY = (
            frameHeight * 0.10
        )

        horizontal = ""
        vertical = ""

        intensity = 1

        if faceX < centerX - toleranceX:

            horizontal = "left"

            distance = abs(
                faceX - centerX
            )

            intensity = max(
                1,
                int(distance / 50)
            )

        elif faceX > centerX + toleranceX:

            horizontal = "right"

            distance = abs(
                faceX - centerX
            )

            intensity = max(
                1,
                int(distance / 50)
            )

        if faceY < centerY - toleranceY:

            vertical = "down"

        elif faceY > centerY + toleranceY:

            vertical = "up"

        return (
            horizontal,
            vertical,
            intensity
        )

    # =====================================================
    # BLUR
    # =====================================================

    def isBlurry(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        variance = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        return variance < 35

    # =====================================================
    # LIGHTING
    # =====================================================

    def analyzeLighting(
        self,
        gray,
        x,
        y,
        w,
        h
    ):

        frameBrightness = gray.mean()

        faceRegion = gray[
            y:y + h,
            x:x + w
        ]

        faceBrightness = (
            faceRegion.mean()
        )

        feedback = []

        if faceBrightness < 60:

            feedback.append(
                "Face too dark"
            )

        elif faceBrightness > 190:

            feedback.append(
                "Face too bright"
            )

        if frameBrightness < 40:

            feedback.append(
                "Room very dark"
            )

        elif frameBrightness > 210:

            feedback.append(
                "Room very bright"
            )

        return feedback

    # =====================================================
    # COUNTDOWN
    # =====================================================

    def countdownSpeech(self):

        ui.message("3")
        time.sleep(1)

        ui.message("2")
        time.sleep(1)

        ui.message("1")
        time.sleep(1)

    # =====================================================
    # CAPTURE
    # =====================================================

    def captureSelfie(self, frame):

        if self.isBlurry(frame):

            ui.message(
                "Image blurry. Hold still"
            )

            return False

        picturesFolder = os.path.join(
            os.path.expanduser("~"),
            "Pictures"
        )

        filename = time.strftime(
            "Selfie_%d_%m_%Y.jpg"
        )

        savePath = os.path.join(
            picturesFolder,
            filename
        )

        counter = 1

        while os.path.exists(
            savePath
        ):

            filename = time.strftime(
                f"Selfie_%d_%m_%Y_{counter}.jpg"
            )

            savePath = os.path.join(
                picturesFolder,
                filename
            )

            counter += 1

        cv2.imwrite(
            savePath,
            frame
        )

        self.sendFeedback(
            "capture",
            "Sharp selfie captured"
        )

        return True

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    def cameraLoop(self):

        self.cap = cv2.VideoCapture(
            0
        )

        if not self.cap.isOpened():

            ui.message(
                "Unable to access camera"
            )

            self.running = False
            return

        cascadePath = os.path.join(
            os.path.dirname(__file__),
            "haarcascade_frontalface_default.xml"
        )

        faceCascade = cv2.CascadeClassifier(
            cascadePath
        )

        if faceCascade.empty():

            ui.message(
                "Face cascade failed to load"
            )

            self.running = False

            if self.cap:
                self.cap.release()
                self.cap = None

            return

        while self.running:

            ret, frame = self.cap.read()

            if not ret:

                ui.message(
                    "Camera frame failed"
                )

                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = (
                faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(40, 40)
                )
            )

            if len(faces) == 0:

                self.sendFeedback(
                    "noface"
                )

                self.centerStartTime = None
                self.centerConfirmed = False

                time.sleep(0.1)

                continue

            x, y, w, h = max(
                faces,
                key=lambda f:
                f[2] * f[3]
            )

            frameWidth = (
                frame.shape[1]
            )

            frameHeight = (
                frame.shape[0]
            )

            faceCenterX = (
                x + (w // 2)
            )

            faceCenterY = (
                y + (h // 2)
            )

            (
                horizontal,
                vertical,
                intensity
            ) = (
                self.getDirectionFeedback(
                    faceCenterX,
                    faceCenterY,
                    frameWidth,
                    frameHeight
                )
            )

            centered = (
                horizontal == ""
                and
                vertical == ""
            )

            if centered:

                if self.centerStartTime is None:

                    self.centerStartTime = (
                        time.time()
                    )

                stableDuration = (
                    time.time()
                    - self.centerStartTime
                )

                if (
                    stableDuration > 0.7
                    and not self.centerConfirmed
                ):

                    self.sendFeedback(
                        "centered"
                    )

                    self.centerConfirmed = True

            else:

                self.centerStartTime = None
                self.centerConfirmed = False

            if not centered:

                if horizontal:

                    self.sendFeedback(
                        "direction",
                        horizontal,
                        intensity
                    )

                elif vertical:

                    self.sendFeedback(
                        "direction",
                        vertical,
                        intensity
                    )

            if self.settings[
                "distanceFeedback"
            ]:

                faceArea = w * h

                frameArea = (
                    frameWidth
                    * frameHeight
                )

                ratio = (
                    faceArea
                    / frameArea
                )

                distanceGood = (
                    0.08 <= ratio <= 0.30
                )

                if (
                    centered
                    and not distanceGood
                ):

                    if ratio < 0.08:

                        self.sendFeedback(
                            "distance",
                            "Move closer"
                        )

                    else:

                        self.sendFeedback(
                            "distance",
                            "Move farther"
                        )

            else:

                distanceGood = True

            lightingGood = True

            if self.settings[
                "lightingFeedback"
            ]:

                lightingFeedback = (
                    self.analyzeLighting(
                        gray,
                        x,
                        y,
                        w,
                        h
                    )
                )

                if lightingFeedback:

                    lightingGood = False

                if centered:

                    for item in lightingFeedback:

                        self.sendFeedback(
                            "lighting",
                            item
                        )

            ready = (
                centered
                and distanceGood
                and lightingGood
            )

            if (
                self.selfieMode
                and ready
                and not self.captureInProgress
            ):

                self.captureInProgress = True

                self.sendFeedback(
                    "capture",
                    "Perfect position. Hold still"
                )

                self.countdownSpeech()

                self.captureSelfie(
                    frame
                )

                self.captureInProgress = False

                time.sleep(2)

            time.sleep(0.1)

        if self.cap:

            self.cap.release()

        self.cap = None
        self.running = False

    # =====================================================
    # GESTURES
    # =====================================================

    __gestures = {

        "kb:NVDA+Shift+G":
            "toggleCamera",

        "kb:NVDA+Shift+C":
            "toggleSelfieMode",

        "kb:NVDA+Shift+1":
            "setDirectionMode",

        "kb:NVDA+Shift+2":
            "setClockMode",

        "kb:NVDA+Shift+S":
            "setSpeechMode",

        "kb:NVDA+Shift+T":
            "setToneMode",

        "kb:NVDA+Shift+B":
            "setSpeechToneMode"
    }

    # =====================================================
    # TOGGLE CAMERA
    # =====================================================

    @scriptHandler.script(
        description=
        "Start or stop face guide"
    )

    def script_toggleCamera(
        self,
        gesture
    ):

        if self.running:

            self.stopCamera()

        else:

            self.startCamera()

    # =====================================================
    # SELFIE MODE
    # =====================================================

    @scriptHandler.script(
        description=
        "Toggle selfie mode"
    )

    def script_toggleSelfieMode(
        self,
        gesture
    ):

        self.selfieMode = (
            not self.selfieMode
        )

        if self.selfieMode:

            ui.message(
                "Selfie mode enabled"
            )

        else:

            ui.message(
                "Selfie mode disabled"
            )

    # =====================================================
    # GUIDANCE STYLE
    # =====================================================

    @scriptHandler.script(
        description=
        "Direction guidance mode"
    )

    def script_setDirectionMode(
        self,
        gesture
    ):

        self.settings[
            "guidanceStyle"
        ] = "direction"

        self.saveSettings()

        ui.message(
            "Direction guidance mode"
        )

    @scriptHandler.script(
        description=
        "Clock guidance mode"
    )

    def script_setClockMode(
        self,
        gesture
    ):

        self.settings[
            "guidanceStyle"
        ] = "clock"

        self.saveSettings()

        ui.message(
            "Clock guidance mode"
        )

    # =====================================================
    # FEEDBACK MODES
    # =====================================================

    @scriptHandler.script(
        description=
        "Speech feedback mode"
    )

    def script_setSpeechMode(
        self,
        gesture
    ):

        self.settings[
            "feedbackMode"
        ] = "speech"

        self.saveSettings()

        ui.message(
            "Speech feedback mode"
        )

    @scriptHandler.script(
        description=
        "Tone feedback mode"
    )

    def script_setToneMode(
        self,
        gesture
    ):

        self.settings[
            "feedbackMode"
        ] = "tone"

        self.saveSettings()

        ui.message(
            "Tone feedback mode"
        )

    @scriptHandler.script(
        description=
        "Speech and tone feedback mode"
    )

    def script_setSpeechToneMode(
        self,
        gesture
    ):

        self.settings[
            "feedbackMode"
        ] = "speechAndTone"

        self.saveSettings()

        ui.message(
            "Speech and tone feedback mode"
        )