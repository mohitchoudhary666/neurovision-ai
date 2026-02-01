from flask import Flask, render_template, Response, request, jsonify
from real_time_computer_vision_system import VisionSystem
import threading
import time

app = Flask(__name__)
vision = VisionSystem()
vision.web_mode = True  # Set web mode flag

# Store real-time stats
stats = {
    'face_count': 0,
    'fps': 0,
    'current_mode': 'color',
    'capture_status': 'ready'
}

def update_stats():
    """Update stats from vision system"""
    while True:
        stats['face_count'] = vision.face_count
        stats['fps'] = vision.fps
        stats['current_mode'] = vision.mode
        time.sleep(0.5)

# Start stats update thread
stats_thread = threading.Thread(target=update_stats, daemon=True)
stats_thread.start()

def generate_frames():
    while True:
        frame = vision.get_processed_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    if mode in ["color", "face", "motion", "capture"]:
        vision.mode = mode
        stats['current_mode'] = mode
    return ("", 204)

@app.route('/capture_face', methods=['POST'])
def capture_face():
    """Capture face from current frame"""
    try:
        # Get a fresh frame
        ret, frame = vision.cap.read()
        if ret and frame is not None:
            success = vision.capture_current_face(frame.copy())
            if success:
                stats['face_count'] = vision.face_count
                stats['capture_status'] = 'success'
                return jsonify({
                    'success': True,
                    'message': 'Face captured successfully!',
                    'face_count': vision.face_count
                })
            else:
                stats['capture_status'] = 'failed'
                return jsonify({
                    'success': False,
                    'message': 'Capture failed. Make sure exactly one clear face is visible.'
                })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to capture frame from camera.'
            })
    except Exception as e:
        print(f"Capture error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'System error: {str(e)}'
        })

@app.route('/get_stats')
def get_stats():
    """Get current system statistics"""
    return jsonify({
        'face_count': vision.face_count,
        'fps': vision.fps,
        'current_mode': vision.mode,
        'show_help': vision.show_help,
        'show_fps': vision.show_fps
    })

if __name__ == "__main__":
    app.run(debug=True, threaded=True)