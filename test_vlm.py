# from openai import OpenAI

# client = OpenAI(
#     base_url="http://promaxgb10-d473.eecs.umich.edu:8000/v1",
#     api_key="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
# )

# response = client.chat.completions.create(
# model="openai/gpt-oss-120b",
# messages=[
# {"role": "user", "content": "Hello"}
# ]
# )

# print(response.choices[0].message.content)
from openai import OpenAI

client = OpenAI( base_url="http://promaxgb10-d473.eecs.umich.edu:8000/v1",
    api_key="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a" )

models = client.models.list()
for m in models.data:
    print(m.id)