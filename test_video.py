from model import analyze_video
import sys

try:
    print(analyze_video('dummy.mp4'))
except Exception as e:
    print("ERROR:", e)
    import traceback
    traceback.print_exc()
