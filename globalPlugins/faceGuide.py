import os
import sys
import threading
import time

addonPath = os.path.dirname(__file__)

# Load bundled libraries
sys.path.insert(
    0,
    os.path.join(addonPath, "lib")
)

import cv2
import numpy as np

import globalPluginHandler
import scriptHandler
import ui


class GlobalPlugin(globalPluginHandler.GlobalPlugin):

    VERSION = "4.0"

    def __init__(self):
        super().__init__()

        self.running = False
        self.selfieMode = False

        self.cap = None
        self.thread = None

        self.captureInProgress = False
        self.lastCaptureTime = 0

        self.smileStartTime = None

    def terminate(self):

        self.stopCamera()
        super().terminate()

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

    def stopCamera(self):

        self.running = False

        self.captureInProgress = False

        self.smileStartTime = None

        if self.cap:

            self.cap.release()
            self.cap = None

        ui.message(
            "Face guide stopped"
        )

    def getClockDirection(
        self,
        faceX,
        faceY,
        frameWidth,
        frameHeight
    ):

        xRatio = faceX / frameWidth
        yRatio = faceY / frameHeight

        if (
            0.4 <= xRatio <= 0.6
            and
            0.4 <= yRatio <= 0.6
        ):

            return "Face at center"

        if yRatio < 0.35:

            if xRatio < 0.4:

                return "Face at 2 o'clock"

            elif xRatio > 0.6:

                return "Face at 10 o'clock"

            else:

                return "Face at 12 o'clock"

        elif yRatio > 0.65:

            if xRatio < 0.4:

                return "Face at 4 o'clock"

            elif xRatio > 0.6:

                return "Face at 8 o'clock"

            else:

                return "Face at 6 o'clock"

        else:

            if xRatio < 0.4:

                return "Face at 3 o'clock"

            else:

                return "Face at 9 o'clock"

    def countdownSpeech(self):

        ui.message("3")
        time.sleep(1)

        ui.message("2")
        time.sleep(1)

        ui.message("1")
        time.sleep(1)

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

    def analyzeLighting(
        self,
        gray,
        x,
        y,
        w,
        h
    ):

        frameBrightness = np.mean(gray)

        faceRegion = gray[
            y:y + h,
            x:x + w
        ]

        faceBrightness = np.mean(faceRegion)

        mask = np.ones(
            gray.shape,
            dtype=np.uint8
        ) * 255

        mask[
            y:y + h,
            x:x + w
        ] = 0

        backgroundPixels = gray[
            mask == 255
        ]

        backgroundBrightness = np.mean(
            backgroundPixels
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

        brightnessDifference = (
            backgroundBrightness
            - faceBrightness
        )

        if brightnessDifference > 40:

            feedback.append(
                "Bright background behind you"
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

        while os.path.exists(savePath):

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

        ui.message(
            "Sharp selfie captured"
        )

        return True

    def cameraLoop(self):

        try:

            ui.message(
                "Opening camera"
            )

            self.cap = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

            if not self.cap.isOpened():

                ui.message(
                    "Unable to access camera"
                )

                self.running = False
                return

            ui.message(
                "Camera connected"
            )

            faceCascadePath = os.path.join(
                addonPath,
                "haarcascade_frontalface_default.xml"
            )

            smileCascadePath = os.path.join(
                addonPath,
                "haarcascade_smile.xml"
            )

            faceCascade = cv2.CascadeClassifier(
                faceCascadePath
            )

            smileCascade = cv2.CascadeClassifier(
                smileCascadePath
            )

            if faceCascade.empty():

                ui.message(
                    "Face detection file missing"
                )

                self.running = False
                return

            if smileCascade.empty():

                ui.message(
                    "Smile detection file missing"
                )

                self.running = False
                return

            lastMessage = ""

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

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )

                if len(faces) > 0:

                    x, y, w, h = max(
                        faces,
                        key=lambda f: f[2] * f[3]
                    )

                    faceGray = gray[
                        y:y+h,
                        x:x+w
                    ]

                    smiles = smileCascade.detectMultiScale(
                        faceGray,
                        scaleFactor=1.8,
                        minNeighbors=25,
                        minSize=(40, 40)
                    )

                    smileDetected = False

                    for (
                        sx,
                        sy,
                        sw,
                        sh
                    ) in smiles:

                        smileRatio = sw / w

                        if smileRatio > 0.35:

                            smileDetected = True
                            break

                    frameWidth = frame.shape[1]
                    frameHeight = frame.shape[0]

                    faceCenterX = x + (w // 2)
                    faceCenterY = y + (h // 2)

                    direction = self.getClockDirection(
                        faceCenterX,
                        faceCenterY,
                        frameWidth,
                        frameHeight
                    )

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

                    lightingFeedback = (
                        self.analyzeLighting(
                            gray,
                            x,
                            y,
                            w,
                            h
                        )
                    )

                    parts = []

                    if direction != "Face at center":

                        parts.append(direction)

                    if not distanceGood:

                        if ratio < 0.08:

                            parts.append(
                                "Move closer"
                            )

                        else:

                            parts.append(
                                "Move farther"
                            )

                    parts.extend(
                        lightingFeedback
                    )

                    if not parts:

                        if smileDetected:

                            message = (
                                "Smile detected"
                            )

                            if (
                                self.smileStartTime
                                is None
                            ):

                                self.smileStartTime = (
                                    time.time()
                                )

                        else:

                            message = (
                                "Face centered"
                            )

                            self.smileStartTime = None

                        currentTime = time.time()

                        stableSmile = False

                        if (
                            self.smileStartTime
                            is not None
                        ):

                            if (
                                currentTime
                                - self.smileStartTime
                                >= 2
                            ):

                                stableSmile = True

                        if (
                            self.selfieMode
                            and
                            stableSmile
                            and
                            not self.captureInProgress
                            and
                            currentTime
                            - self.lastCaptureTime
                            > 5
                        ):

                            self.captureInProgress = True

                            ui.message(
                                "Perfect smile detected"
                            )

                            self.countdownSpeech()

                            ret, freshFrame = (
                                self.cap.read()
                            )

                            if ret:

                                success = (
                                    self.captureSelfie(
                                        freshFrame
                                    )
                                )

                                if success:

                                    self.lastCaptureTime = (
                                        time.time()
                                    )

                                    ui.message(
                                        "Selfie saved"
                                    )

                            self.captureInProgress = False

                            self.smileStartTime = None

                    else:

                        message = ", ".join(parts)

                        self.smileStartTime = None

                    if message != lastMessage:

                        ui.message(message)

                        lastMessage = message

                else:

                    if (
                        lastMessage
                        != "No face detected"
                    ):

                        ui.message(
                            "No face detected"
                        )

                        lastMessage = (
                            "No face detected"
                        )

                    self.smileStartTime = None

                time.sleep(0.7)

        except Exception as e:

            ui.message(
                f"Face guide error {str(e)}"
            )

        finally:

            if self.cap:

                self.cap.release()

            self.cap = None

            self.running = False

    __gestures = {
        "kb:NVDA+Shift+G":
            "toggleCamera",

        "kb:NVDA+Shift+C":
            "toggleSelfieMode"
    }

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

    @scriptHandler.script(
        description=
        "Toggle smart selfie mode"
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
                "Smart selfie mode enabled"
            )

        else:

            ui.message(
                "Smart selfie mode disabled"
            )