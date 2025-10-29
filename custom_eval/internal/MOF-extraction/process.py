import json

with open("raw.json", "r", encoding="utf8") as f:
    data = json.load(f)

output = []

for k, value in data.items():
    count = 0
    for v in value:
        output_item = {}
        output_item['file ID'] = str(v['file ID']) + f"-{count}"
        output_item['file name short'] = v['file name short']
        output_item['content'] = v['content']
        output_item['doi'] = v['doi']
        output_item['mof_gt'] = v['mof_gt']
        output.append(output_item)
        count += 1

with open("default_test.jsonl", "w", encoding="utf8") as wf:
    for output_item in output:
        wf.write(json.dumps(output_item)+"\n")
