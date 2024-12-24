import json

with open("playground/data/llava_v1_5_mix665k.json",'r',encoding='UTF-8') as f:
    data = json.load(f)
    print(len(data))

data1 = data[::10]
print(len(data1))
with open("playground/data/llava_v1_5_mix67k.json", "w", encoding="utf-8") as f:
    json.dump(data1, f)