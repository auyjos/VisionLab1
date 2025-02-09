import cv2
import numpy as np


class ColorDetector:
    def __init__(self):
        """Inicializa la captura de video y el diccionario de colores."""
        self.cap = cv2.VideoCapture(0)
        self.colors = self.defineColors()

    def defineColors(self):
        """Define los colores primarios y secundarios con sus rangos en HSV."""
        return {
            "rojo1": {"lower": np.array([0, 120, 70]), "upper": np.array([10, 255, 255]), "color": (0, 0, 255)},
            "rojo2": {"lower": np.array([170, 120, 70]), "upper": np.array([180, 255, 255]), "color": (0, 0, 255)},
            "azul": {"lower": np.array([100, 150, 0]), "upper": np.array([140, 255, 255]), "color": (255, 0, 0)},
            "amarillo": {"lower": np.array([15, 80, 80]), "upper": np.array([35, 255, 255]), "color": (0, 255, 255)},
            "verde": {"lower": np.array([40, 40, 40]), "upper": np.array([80, 255, 255]), "color": (0, 255, 0)},
            "naranja": {"lower": np.array([5, 100, 100]), "upper": np.array([15, 255, 255]), "color": (0, 165, 255)},
            "morado": {"lower": np.array([130, 50, 50]), "upper": np.array([160, 255, 255]), "color": (128, 0, 128)}
        }

    def processFrame(self, frame):
        """Convierte el frame a HSV y detecta colores definidos."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for colorName, colorInfo in self.colors.items():
            lower = colorInfo["lower"]
            upper = colorInfo["upper"]
            colorBgr = colorInfo["color"]

            mask = cv2.inRange(hsv, lower, upper)
            mask = self.applyMorphology(mask)

            self.detectContours(frame, mask, colorBgr, colorName)

    def applyMorphology(self, mask):
        """Aplica filtros morfológicos para mejorar la máscara."""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def detectContours(self, frame, mask, colorBgr, colorName):
        """Detecta y dibuja contornos en el frame."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(frame, [contour], -1, colorBgr, 2)
                self.drawCenterPoint(frame, contour, colorBgr)
                self.drawLabel(frame, contour, colorName, colorBgr)

    def drawCenterPoint(self, frame, contour, colorBgr):
        """Calcula y dibuja el centro del objeto detectado."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, colorBgr, -1)

    def drawLabel(self, frame, contour, colorName, colorBgr):
        """Dibuja una etiqueta con el nombre del color detectado."""
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(frame, colorName, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorBgr, 2)

    def run(self):
        """Ejecuta la detección de colores en tiempo real."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.processFrame(frame)
            cv2.imshow('Deteccion de colores', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ColorDetector()
    detector.run()
