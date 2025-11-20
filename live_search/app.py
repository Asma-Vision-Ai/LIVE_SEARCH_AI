from flask import Flask, render_template, request, jsonify, send_file, Response
import torch
import open_clip
from PIL import Image
import cv2
import numpy as np
import base64
from io import BytesIO
import os
from werkzeug.utils import secure_filename
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import threading
import time
from collections import deque

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Email Configuration
EMAIL_CONFIG = {
    'SMTP_SERVER': 'smtp.gmail.com',
    'SMTP_PORT': 587,
    'SENDER_EMAIL': 'rjn0047@gmail.com',
    'SENDER_PASSWORD': 'bsog okrj zwxd zhxk',
    'RECIPIENT_EMAIL': 'rjn1032@gmail.com'
}

# Alert keywords that trigger email notifications
ALERT_KEYWORDS = [
    'violence', 'violent', 'fight', 'fighting', 'punch', 'kick', 'attack',
    'fire', 'flame', 'smoke', 'burning', 'explosion',
    'suspicious', 'weapon', 'gun', 'knife', 'robbery', 'theft', 'thief',
    'accident', 'crash', 'emergency', 'danger', 'threat'
]

# Alert threshold - minimum similarity score to trigger alert
ALERT_THRESHOLD = 0.25

# Limit how many images we attach to an email
MAX_ATTACHMENTS = 5

# RTSP Stream Configuration
RTSP_STREAMS = {}  # {stream_id: stream_config}
STREAM_THREADS = {}  # {stream_id: thread}
ALERT_COOLDOWN = 60  # seconds between alerts for same stream

# Load CLIP model
print("Loading CLIP model...")
try:
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)
    model.eval()
    print(f"CLIP model loaded on {DEVICE}")
except Exception as e:
    print("Failed to load CLIP model:", e)
    raise

# Try to load SAM model
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
    print("Loading SAM model...")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("SAM model loaded successfully!")
except Exception as e:
    SAM_AVAILABLE = False
    print("SAM not available:", e)


class RTSPStreamMonitor:
    def __init__(self, stream_id, rtsp_url, queries, check_interval=2):
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.queries = queries  # List of queries to monitor
        self.check_interval = check_interval  # seconds between checks
        self.running = False
        self.thread = None
        self.last_alert_time = {}  # {query: timestamp}
        self.latest_frame = None
        self.latest_results = {}
        
    def start(self):
        if self.running:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._monitor_stream, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
    def _monitor_stream(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"Failed to open RTSP stream: {self.rtsp_url}")
                return
            
            print(f"Monitoring stream {self.stream_id}: {self.rtsp_url}")
            frame_count = 0
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print(f"Stream {self.stream_id}: Failed to read frame, reconnecting...")
                    cap.release()
                    time.sleep(5)
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
                
                frame_count += 1
                
                # Store latest frame for streaming
                self.latest_frame = frame.copy()
                
                # Check at specified interval
                if frame_count % (int(30 * self.check_interval)) == 0:  # Assuming 30fps
                    self._analyze_frame(frame)
                
        except Exception as e:
            print(f"Error monitoring stream {self.stream_id}: {e}")
        finally:
            if cap:
                cap.release()
    
    def _analyze_frame(self, frame):
        try:
            # Convert to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Check each query
            for query in self.queries:
                # Get CLIP embeddings
                image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
                text_tensor = tokenizer([query]).to(DEVICE)
                
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    text_features = model.encode_text(text_tensor)
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (image_features @ text_features.T).squeeze().item()
                
                # Store result
                self.latest_results[query] = {
                    'score': similarity,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Check if alert should be triggered
                if similarity >= ALERT_THRESHOLD and is_alert_query(query):
                    current_time = time.time()
                    last_alert = self.last_alert_time.get(query, 0)
                    
                    # Only send alert if cooldown period has passed
                    if current_time - last_alert >= ALERT_COOLDOWN:
                        self._send_alert(query, similarity, pil_image)
                        self.last_alert_time[query] = current_time
                        print(f"Alert sent for stream {self.stream_id}, query: {query}, score: {similarity:.3f}")
        
        except Exception as e:
            print(f"Error analyzing frame for stream {self.stream_id}: {e}")
    
    def _send_alert(self, query, score, image):
        try:
            # Apply SAM if available
            bboxes = apply_sam_segmentation(image) if SAM_AVAILABLE else []
            
            # Convert to base64
            frame_base64 = image_to_base64(image, bboxes)
            
            # Prepare alert info
            alert_info = {
                'stream_id': self.stream_id,
                'rtsp_url': self.rtsp_url,
                'query': query,
                'score': score,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'objects_detected': len(bboxes)
            }
            
            # Send email
            send_rtsp_alert(alert_info, [frame_base64])
            
        except Exception as e:
            print(f"Error sending alert for stream {self.stream_id}: {e}")
    
    def get_latest_frame_jpeg(self):
        if self.latest_frame is None:
            return None
        ret, buffer = cv2.imencode('.jpg', self.latest_frame)
        return buffer.tobytes()


def send_email_alert(video_info, image_base64_list=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['SENDER_EMAIL']
        msg['To'] = EMAIL_CONFIG['RECIPIENT_EMAIL']
        msg['Subject'] = "Video Alert Notification"

        body = (
            "Video Alert Triggered\n\n"
            f"Duration: {video_info.get('duration')}\n"
            f"FPS: {video_info.get('fps')}\n"
            f"Total Frames: {video_info.get('total_frames')}\n"
            f"Frames Extracted: {video_info.get('frames_extracted')}\n"
            f"Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        msg.attach(MIMEText(body, 'plain'))

        if image_base64_list:
            for idx, img64 in enumerate(image_base64_list[:MAX_ATTACHMENTS], start=1):
                try:
                    if ',' in img64:
                        img_data = img64.split(",")[1]
                    else:
                        img_data = img64
                    img_bytes = base64.b64decode(img_data)
                    image_part = MIMEImage(img_bytes)
                    filename = f"frame_{idx}.jpg"
                    image_part.add_header('Content-Disposition', 'attachment', filename=filename)
                    msg.attach(image_part)
                except Exception as ie:
                    print("Image attach error:", ie)

        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'])
        server.starttls()
        server.login(EMAIL_CONFIG['SENDER_EMAIL'], EMAIL_CONFIG['SENDER_PASSWORD'])
        server.send_message(msg)
        server.quit()

        print("Alert email sent successfully.")
        return True

    except Exception as e:
        print("Error sending email:", e)
        return False


def send_rtsp_alert(alert_info, image_base64_list=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['SENDER_EMAIL']
        msg['To'] = EMAIL_CONFIG['RECIPIENT_EMAIL']
        msg['Subject'] = f"LIVE STREAM ALERT - {alert_info['query'].upper()}"

        body = (
            "LIVE STREAM ALERT TRIGGERED\n\n"
            f"Stream ID: {alert_info['stream_id']}\n"
            f"RTSP URL: {alert_info['rtsp_url']}\n"
            f"Detection: {alert_info['query']}\n"
            f"Confidence Score: {alert_info['score']:.3f}\n"
            f"Objects Detected: {alert_info['objects_detected']}\n"
            f"Alert Time: {alert_info['timestamp']}\n\n"
            "Please review the attached frame immediately.\n"
        )
        msg.attach(MIMEText(body, 'plain'))

        if image_base64_list:
            for idx, img64 in enumerate(image_base64_list[:MAX_ATTACHMENTS], start=1):
                try:
                    if ',' in img64:
                        img_data = img64.split(",")[1]
                    else:
                        img_data = img64
                    img_bytes = base64.b64decode(img_data)
                    image_part = MIMEImage(img_bytes)
                    filename = f"alert_frame_{idx}.jpg"
                    image_part.add_header('Content-Disposition', 'attachment', filename=filename)
                    msg.attach(image_part)
                except Exception as ie:
                    print("Image attach error:", ie)

        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'])
        server.starttls()
        server.login(EMAIL_CONFIG['SENDER_EMAIL'], EMAIL_CONFIG['SENDER_PASSWORD'])
        server.send_message(msg)
        server.quit()

        print("RTSP alert email sent successfully.")
        return True

    except Exception as e:
        print("Error sending RTSP alert email:", e)
        return False


def is_alert_query(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ALERT_KEYWORDS)


def extract_frames(video_path, frames_per_second=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    frame_ids = []
    frame_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    frame_interval = max(1, int(round((fps / frames_per_second)))) if frames_per_second > 0 else 1

    video_info = {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'frame_interval': frame_interval,
        'frames_per_second': frames_per_second
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
            timestamp = frame_count / fps if fps > 0 else 0.0
            frame_ids.append(timestamp)

        frame_count += 1

    cap.release()
    return frames, frame_ids, video_info


def get_clip_embeddings(frames, query, batch_size=32):
    text_tensor = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tensor)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_image_features = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        image_tensors = torch.stack([preprocess(f) for f in batch_frames]).to(DEVICE)
        with torch.no_grad():
            batch_features = model.encode_image(image_tensors)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            all_image_features.append(batch_features.cpu())

    if len(all_image_features) > 0:
        image_features = torch.cat(all_image_features, dim=0)
    else:
        image_features = torch.empty((0, model.visual.output_dim))

    return image_features, text_features.cpu()


def apply_sam_segmentation(image):
    if not SAM_AVAILABLE:
        return []

    image_np = np.array(image)
    try:
        masks = mask_generator.generate(image_np)
    except Exception as e:
        print("SAM generation error:", e)
        return []

    bboxes = []
    for mask in masks:
        bbox = mask.get('bbox', None)
        area = mask.get('area', 0)
        if bbox and area > 1000:
            bboxes.append(bbox)
    return bboxes


def image_to_base64(image, bboxes=None):
    img_np = np.array(image)

    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(img_np, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    img_pil = Image.fromarray(img_np)
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


@app.route('/')
def index():
    if os.path.exists(os.path.join(app.root_path, 'templates', 'index.html')):
        return render_template('index.html')
    return (
        "Video CLIP search API with RTSP Stream Support\n\n"
        "Endpoints:\n"
        "POST /search - Search uploaded video\n"
        "POST /configure-email - Update email settings\n"
        "POST /rtsp/start - Start monitoring RTSP stream\n"
        "POST /rtsp/stop - Stop monitoring RTSP stream\n"
        "GET /rtsp/status - Get status of all streams\n"
        "GET /rtsp/stream/<stream_id> - View live stream\n"
    )


@app.route('/search', methods=['POST'])
def search():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        query = request.form.get('query', '')
        frames_per_second = int(request.form.get('fps', 5))
        top_k = int(request.form.get('top_k', 5))
        send_alert = request.form.get('send_alert', 'true').lower() == 'true'

        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        if not query:
            return jsonify({'error': 'No search query provided'}), 400

        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        frames, frame_times, video_info = extract_frames(video_path, frames_per_second)

        if len(frames) == 0:
            try:
                os.remove(video_path)
            except Exception:
                pass
            return jsonify({'error': 'No frames extracted from video'}), 400

        image_features, text_features = get_clip_embeddings(frames, query)
        if image_features.shape[0] == 0:
            try:
                os.remove(video_path)
            except Exception:
                pass
            return jsonify({'error': 'No image features computed'}), 500

        similarity = (image_features @ text_features.T).squeeze()

        k = min(top_k, similarity.shape[0])
        topk = similarity.topk(k)

        results = []
        alert_triggered = False

        for score, frame_idx in zip(topk.values, topk.indices):
            idx = int(frame_idx.item())
            timestamp = frame_times[idx]
            score_val = float(score.item())

            frame = frames[idx]
            bboxes = apply_sam_segmentation(frame) if SAM_AVAILABLE else []

            frame_base64 = image_to_base64(frame, bboxes)

            results.append({
                'timestamp': timestamp,
                'formatted_time': format_timestamp(timestamp),
                'score': score_val,
                'image': frame_base64,
                'objects_detected': len(bboxes)
            })

            if score_val >= ALERT_THRESHOLD and is_alert_query(query):
                alert_triggered = True

        video_info_out = {
            'duration': video_info['duration'],
            'fps': video_info['fps'],
            'total_frames': video_info['total_frames'],
            'frames_extracted': len(frames)
        }

        email_sent = False
        if alert_triggered or send_alert:
            image_base64_list = [r['image'] for r in results][:MAX_ATTACHMENTS]
            email_sent = send_email_alert(video_info_out, image_base64_list)

        try:
            os.remove(video_path)
        except Exception as e:
            print("Could not remove uploaded video:", e)

        return jsonify({
            'success': True,
            'results': results,
            'video_info': video_info_out,
            'sam_available': SAM_AVAILABLE,
            'alert_triggered': alert_triggered,
            'email_sent': email_sent
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rtsp/start', methods=['POST'])
def start_rtsp_stream():
    """
    Start monitoring an RTSP stream.
    JSON body: {
        "stream_id": "camera1",
        "rtsp_url": "rtsp://username:password@ip:port/stream",
        "queries": ["fighting", "fire", "suspicious activity", "thief"],
        "check_interval": 2
    }
    """
    try:
        data = request.json
        stream_id = data.get('stream_id')
        rtsp_url = data.get('rtsp_url')
        queries = data.get('queries', ['fighting', 'fire', 'suspicious', 'thief'])
        check_interval = data.get('check_interval', 2)
        
        if not stream_id or not rtsp_url:
            return jsonify({'error': 'stream_id and rtsp_url are required'}), 400
        
        # Stop existing stream if running
        if stream_id in RTSP_STREAMS:
            RTSP_STREAMS[stream_id].stop()
        
        # Create and start new monitor
        monitor = RTSPStreamMonitor(stream_id, rtsp_url, queries, check_interval)
        success = monitor.start()
        
        if success:
            RTSP_STREAMS[stream_id] = monitor
            return jsonify({
                'success': True,
                'message': f'Stream {stream_id} started successfully',
                'stream_id': stream_id,
                'queries': queries
            })
        else:
            return jsonify({'error': 'Failed to start stream'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rtsp/stop', methods=['POST'])
def stop_rtsp_stream():
    """
    Stop monitoring an RTSP stream.
    JSON body: {"stream_id": "camera1"}
    """
    try:
        data = request.json
        stream_id = data.get('stream_id')
        
        if not stream_id:
            return jsonify({'error': 'stream_id is required'}), 400
        
        if stream_id in RTSP_STREAMS:
            RTSP_STREAMS[stream_id].stop()
            del RTSP_STREAMS[stream_id]
            return jsonify({
                'success': True,
                'message': f'Stream {stream_id} stopped successfully'
            })
        else:
            return jsonify({'error': f'Stream {stream_id} not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rtsp/status', methods=['GET'])
def get_rtsp_status():
    """Get status of all monitored streams"""
    try:
        status = {}
        for stream_id, monitor in RTSP_STREAMS.items():
            status[stream_id] = {
                'running': monitor.running,
                'rtsp_url': monitor.rtsp_url,
                'queries': monitor.queries,
                'latest_results': monitor.latest_results,
                'check_interval': monitor.check_interval
            }
        
        return jsonify({
            'success': True,
            'streams': status,
            'total_streams': len(status)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rtsp/stream/<stream_id>')
def view_rtsp_stream(stream_id):
    """View live stream as MJPEG"""
    if stream_id not in RTSP_STREAMS:
        return "Stream not found", 404
    
    def generate():
        monitor = RTSP_STREAMS[stream_id]
        while monitor.running:
            frame_bytes = monitor.get_latest_frame_jpeg()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/configure-email', methods=['POST'])
def configure_email():
    try:
        data = request.json or {}
        EMAIL_CONFIG['SENDER_EMAIL'] = data.get('sender_email', EMAIL_CONFIG['SENDER_EMAIL'])
        EMAIL_CONFIG['SENDER_PASSWORD'] = data.get('sender_password', EMAIL_CONFIG['SENDER_PASSWORD'])
        EMAIL_CONFIG['RECIPIENT_EMAIL'] = data.get('recipient_email', EMAIL_CONFIG['RECIPIENT_EMAIL'])
        return jsonify({'success': True, 'message': 'Email configuration updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)