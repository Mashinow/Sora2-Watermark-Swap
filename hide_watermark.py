import argparse
import os
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import logging
logging.captureWarnings(True)  # False positive


class PATHS:
    base = os.getcwd().replace('\\', '/')
    res = base + '/res/'
    wm = res + 'watermarks/'


W_TYPES = {
    1: PATHS.wm + 'watermark_large.png',
    2: PATHS.wm + 'watermark_medium.png',
    3: PATHS.wm + 'watermark_small.png',
    4: PATHS.wm + 'watermark_wsmall.png',
    5: PATHS.wm + 'watermark_wsmall2.png',
    6: PATHS.wm + 'watermark_wsmall3.png'
}



def detect_watermark_zone(frame, watermark_template, threshold=0.3):
    result = cv2.matchTemplate(frame, watermark_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        h, w = watermark_template.shape[:2]
        return (*max_loc, w, h)
    return None


def apply_watermark_blur(frame, zone, blur_strength=15, radius=20):
    x, y, w, h = zone
    center_x = x + w // 2 - 33
    center_y = y + h // 2
    blur_width = 150
    blur_height = 56

    x1 = max(center_x - blur_width // 2, 0)
    y1 = max(center_y - blur_height // 2, 0)
    x2 = min(center_x + blur_width // 2, frame.shape[1])
    y2 = min(center_y + blur_height // 2, frame.shape[0])

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    # Размытие ROI
    roi_blurred = cv2.GaussianBlur(roi, (0, 0), blur_strength)

    # Создаём маску с закруглёнными углами
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    rect = (0, 0, roi.shape[1], roi.shape[0])

    # Нарисуем прямоугольник со скруглёнными углами вручную
    cv2.rectangle(mask, (radius, 0), (rect[2] - radius, rect[3]), 255, -1)
    cv2.rectangle(mask, (0, radius), (rect[2], rect[3] - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (rect[2] - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, rect[3] - radius), radius, 255, -1)
    cv2.circle(mask, (rect[2] - radius, rect[3] - radius), radius, 255, -1)

    # Преобразуем маску в 3 канала
    mask_3ch = cv2.merge([mask, mask, mask])

    # Смешиваем исходный ROI и размытый по маске
    blended = np.where(mask_3ch == 255, roi_blurred, roi)

    frame[y1:y2, x1:x2] = blended
    return frame


def overlay_image(frame, overlay_img, position, offset=(-68, -15)):
    apply_watermark_blur(frame, position)
    x, y, _, _ = position
    x += offset[0]
    y += offset[1]

    h_overlay, w_overlay = overlay_img.shape[:2]
    h_frame, w_frame = frame.shape[:2]

    # Обрезаем, если выходит за границы кадра
    if x < 0:
        overlay_img = overlay_img[:, -x:]
        w_overlay += x
        x = 0
    if y < 0:
        overlay_img = overlay_img[-y:, :]
        h_overlay += y
        y = 0
    if x + w_overlay > w_frame:
        overlay_img = overlay_img[:, :w_frame - x]
        w_overlay = w_frame - x
    if y + h_overlay > h_frame:
        overlay_img = overlay_img[:h_frame - y, :]
        h_overlay = h_frame - y

    if h_overlay <= 0 or w_overlay <= 0:
        return frame

    if overlay_img.shape[2] == 4:
        overlay_rgb = overlay_img[:, :, :3].astype(np.float32)
        alpha = overlay_img[:, :, 3].astype(np.float32) / 255.0
    else:
        overlay_rgb = overlay_img.astype(np.float32)
        alpha = np.ones((h_overlay, w_overlay), dtype=np.float32)

    roi = frame[y:y+h_overlay, x:x+w_overlay].astype(np.float32)
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    blended = overlay_rgb * alpha_3 + roi * (1 - alpha_3)

    frame[y:y+h_overlay, x:x+w_overlay] = blended.astype(np.uint8)
    return frame


def is_zones_close(a, b, tolerance=3) -> bool:
    return all(abs(a[i] - b[i]) <= tolerance for i in range(4))


def process_video(input_path, watermark_path, overlay_path, output_path):
    cap = cv2.VideoCapture(input_path)
    video_clip = VideoFileClip(input_path)
    frames = []
    audio = video_clip.audio
    fps = cap.get(cv2.CAP_PROP_FPS)
    watermark_template = cv2.imread(watermark_path)
    overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # RGBA
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = {}

    last_pos = (0, 0, 0, 0)
    ident_count = 0
    skip_frames = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if skip_frames > 0:
            positions[i + 1] = last_pos
            skip_frames -= 1
            continue
        zone = detect_watermark_zone(frame, watermark_template)
        if not zone:
            positions[i+1] = (0, 0, 0, 0)
            continue
        positions[i+1] = zone
        if not is_zones_close(last_pos, zone):
            last_pos = zone
            ident_count = 0
        else:
            ident_count += 1
        if ident_count >= 5:
            sw_pos = 67
            next_multiple = ((i + sw_pos) // sw_pos) * sw_pos
            start_idx = max(next_multiple - sw_pos, 0)
            end_idx = min(next_multiple, frame_count)
            for idx in range(start_idx, end_idx):
                positions[idx] = last_pos
            skip_frames = min(next_multiple - i, frame_count - i)
            ident_count = 0
    cap.release()
    cap = cv2.VideoCapture(input_path)
    last_pos = positions[1]
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        cur_pos = positions[i+1]
        frame = overlay_image(frame, overlay_img, cur_pos)
        if last_pos != cur_pos:
            frame = overlay_image(frame, overlay_img, last_pos)
            last_pos = cur_pos
        frames.append(frame)
    cap.release()
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
    if audio is not None:
        clip = clip.with_audio(audio)
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    video_clip.close()
    clip.close()
    if audio is not None:
        audio.close()


def detect_best_watermark_type(input_path):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(15, total_frames)  # анализируем первые 15 кадров
    scores = {t: [] for t in W_TYPES}

    for i in range(sample_count):
        ret, frame = cap.read()
        if not ret:
            break
        for t, path in W_TYPES.items():
            tmpl = cv2.imread(path)
            result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            scores[t].append(max_val)

    cap.release()
    avg_scores = {t: np.mean(v) if len(v) > 0 else 0 for t, v in scores.items()}
    best_type = max(avg_scores, key=avg_scores.get)

    # если все средние значения ниже 0.5 — выбираем первую (large)
    if all(val < 0.5 for val in avg_scores.values()):
        best_type = 1

    print(f"[AutoDetect] Selected: {best_type} ({W_TYPES[best_type]}), score={avg_scores[best_type]:.3f}")
    return best_type


def hide_watermark(input_video, output_video, force_type=0):
    if force_type:
        best_type = force_type
    else:
        best_type = detect_best_watermark_type(input_video)

    watermark_img = W_TYPES[best_type]
    overlay_img = os.path.join(PATHS.res, "overlay.png")

    process_video(input_video, watermark_img, overlay_img, output_video)


def main():
    parser = argparse.ArgumentParser(description="Hide watermark from video")
    parser.add_argument("-i", "--input", default='input.mp4', help="Path to input video")
    parser.add_argument("-o", "--output", default="output.mp4", help="Path to output video (default: output.mp4)")
    parser.add_argument("-f", "--force_type", type=int, default=0, help="Force watermark type (default: 0, auto-detect)")
    args = parser.parse_args()
    hide_watermark(args.input, args.output, args.force_type)


if __name__ == "__main__":
    main()
