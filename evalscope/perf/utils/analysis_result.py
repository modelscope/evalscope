import base64
import json
import pickle
import sqlite3

result_db_path = './outputs/qwen2.5_benchmark_20241111_160543.db'
con = sqlite3.connect(result_db_path)
query_sql = "SELECT request, response_messages, prompt_tokens, completion_tokens \
                FROM result WHERE success='1'"

# how to save base64.b64encode(pickle.dumps(benchmark_data["request"])).decode("ascii"),
with con:
    rows = con.execute(query_sql).fetchall()
    if len(rows) > 0:
        for row in rows:
            request = row[0]
            responses = row[1]
            request = base64.b64decode(request)
            request = pickle.loads(request)
            responses = base64.b64decode(responses)
            responses = pickle.loads(responses)
            response_content = ''
            for response in responses:
                response = json.loads(response)
                if not response['choices']:
                    continue
                response_content += response['choices'][0]['delta']['content']
            print('prompt: %s, tokens: %s, completion: %s, tokens: %s' %
                  (request['messages'][0]['content'], row[2], response_content, row[3]))
