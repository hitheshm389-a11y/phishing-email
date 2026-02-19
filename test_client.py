import requests

SAMPLE = '''Subject: Immediate action required\n\nWe detected unusual activity on your account. Please verify at http://fake-login.example.com to avoid suspension.''' 

resp = requests.post('http://127.0.0.1:5000/analyze', json={'email': SAMPLE})
print('Status:', resp.status_code)
print(resp.json())
