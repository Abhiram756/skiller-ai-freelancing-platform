import sys
import json
import base64
import numpy as np

# Try importing OpenCV
try:
    import cv2
except ImportError:
    print(json.dumps({"error": "OpenCV not installed. Run: pip install opencv-python"}))
    sys.exit(1)

def detect_face():
    try:
        # 1. Read Input (Base64 String)
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No image data provided"}))
            return

        request = json.loads(input_data)
        image_data = request.get('image', '')

        # Remove header if present (data:image/jpeg;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # 2. Decode Image
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print(json.dumps({"error": "Failed to decode image"}))
            return

        # 3. Load Haar Cascade (Pre-trained ML Model for Face Detection)
        # We use the default one provided by cv2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to Grayscale for efficiency
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4. Detect Faces (Haar Cascade acts as our primary localization engine)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            clarity_score = round(min(1.0, w / 150.0), 2)
            bboxes = [[int(f[0]), int(f[1]), int(f[2]), int(f[3])] for f in faces]
            
            print(json.dumps({
                "verified": True, 
                "faces_detected": len(faces),
                "bounding_boxes": bboxes,
                "clarity_score": clarity_score,
                "message": "Face detected. Identity verification requires a deep learning recognition model."
            }))
        else:
            # No Face
            print(json.dumps({
                "verified": False, 
                "message": "No Face Detected. Neural scan failed."
            }))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    detect_face()
