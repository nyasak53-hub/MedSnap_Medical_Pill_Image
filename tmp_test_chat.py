import sys
sys.path.insert(0, '.')
import app

client = app.app.test_client()
resp = client.post('/api/chat', json={'messages': [{'role': 'user', 'content': 'hello'}]})
print(resp.status_code)
print(resp.get_json())
