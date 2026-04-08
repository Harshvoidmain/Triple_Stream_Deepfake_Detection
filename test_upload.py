import requests
with open('dummy.mp4', 'rb') as f:
    res = requests.post('http://127.0.0.1:5000/api/analyze', files={'video': f})
    print("STATUS:", res.status_code)
    print("RESPONSE:", res.text)
