import base64
import pickle
import sqlite3

db_path = 'your db path'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取列名
cursor.execute('PRAGMA table_info(result)')
columns = [info[1] for info in cursor.fetchall()]
print('列名：', columns)

cursor.execute('SELECT * FROM result WHERE success=1 AND first_chunk_latency > 1')
rows = cursor.fetchall()
print(f'len(rows): {len(rows)}')

for row in rows:
    row_dict = dict(zip(columns, row))
    # The request is JSON text; only response_messages needs decoding.
    row_dict['response_messages'] = pickle.loads(base64.b64decode(row_dict['response_messages']))
    response = row_dict['response_messages'][0] if row_dict['response_messages'] else None
    response_id = response.get('id') if isinstance(response, dict) else None
    # print(row_dict)
    print(
        f'request_id: {response_id or row_dict["request_id"]}, first_chunk_latency: {row_dict["first_chunk_latency"]}'
    )
    # 如果只想看一个可以break
    # break
