from pathlib import Path

from scripts.logging import logger


def generate_random_hash(length=8):
    import hashlib
    import secrets

    # Generate a random number or string
    random_data = secrets.token_bytes(32)  # 32 bytes of random data

    # Create a SHA-256 hash of the random data
    hash_object = hashlib.sha256(random_data)
    hash_hex = hash_object.hexdigest()

    # Get the first 10 characters
    if length > len(hash_hex):
        length = len(hash_hex)
    return hash_hex[:length]


def get_animatediff_arg(p):
    """
    Get AnimateDiff argument from `p`. If it's a dict, convert it to AnimateDiffProcess.
    """
    if not p.scripts:
        return None

    for script in p.scripts.alwayson_scripts:
        if script.title().lower() == "animatediff":
            animatediff_arg = p.script_args[script.args_from]
            if isinstance(animatediff_arg, dict):
                from scripts.animatediff_ui import AnimateDiffProcess
                animatediff_arg = AnimateDiffProcess(**animatediff_arg)
                p.script_args[script.args_from] = animatediff_arg
            return animatediff_arg

    return None


def ffmpeg_extract_frames(source_video: str, output_dir: str, extract_key: bool = False):
    command = ["ffmpeg"]
    from modules.devices import device
    if "cuda" in str(device):
        command.extend(["-hwaccel", "cuda"])
    command.extend(["-i", source_video])
    if extract_key:
        command.extend(["-vf", "select='eq(pict_type,I)'", "-vsync", "vfr"])
    else:
        command.extend(["-filter:v", "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command.extend(["-qscale:v", "1", "-qmin", "1", "-c:a", "copy", str(output_dir / '%09d.jpg')])
    logger.info(f"[AnimateDiff] Attempting to extract frames via ffmpeg from {source_video} to {output_dir}")
    import subprocess
    subprocess.run(command, check=True)


def cv2_extract_frames(source_video: str, output_dir: str):
    logger.info(f"[AnimateDiff] Attempting to extract frames via OpenCV from {source_video} to {output_dir}")
    import cv2
    cap = cv2.VideoCapture(source_video)
    frame_count = 0
    tmp_frame_dir = Path(output_dir)
    tmp_frame_dir.mkdir(parents=True, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{tmp_frame_dir}/{frame_count}.png", frame)
        frame_count += 1
    cap.release()
    


def extract_frames_from_video(params):
    if not params.video_source:
        return
    from modules.paths import data_path
    from modules import shared
    params.video_path = shared.opts.data.get(
        "animatediff_frame_extract_path",
        f"{data_path}/tmp/animatediff-frames")
    params.video_path += f"{params.video_source}-{generate_random_hash()}"
    try:
        ffmpeg_extract_frames(params.video_source, params.video_path)
    except Exception as e:
        logger.error(f"[AnimateDiff] Error extracting frames via ffmpeg: {e}, attempting OpenCV.")
        cv2_extract_frames(params.video_source, params.video_path)
