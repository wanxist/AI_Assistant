from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="")  # 请填写您自己的 API Key

response = client.chat.completions.create(
    model="glm-5.1",
    messages=[
        {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
        {"role": "assistant", "content": "当然，要创作一个吸引人的口号，请告诉我一些关于您产品的信息"},
        {"role": "user", "content": "智谱开放平台"}
    ],
    # thinking={
    #     "type": "enabled",    # 启用深度思考模式
    # },
    max_tokens=65536,          # 最大输出 tokens
    temperature=0.5          # 控制输出的随机性
)

# 获取完整回复
print(response.choices[0].message)


# from zai import ZhipuAiClient

# client = ZhipuAiClient(api_key="your-api-key")  # 请填写您自己的 API Key

# response = client.chat.completions.create(
#     model="glm-5.1",
#     messages=[
#         {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
#         {"role": "assistant", "content": "当然，要创作一个吸引人的口号，请告诉我一些关于您产品的信息"},
#         {"role": "user", "content": "智谱开放平台"}
#     ],
#     thinking={
#         "type": "enabled",    # 启用深度思考模式
#     },
#     stream=True,              # 启用流式输出
#     max_tokens=65536,          # 最大输出tokens
#     temperature=1.0           # 控制输出的随机性
# )

# # 流式获取回复
# for chunk in response:
#     if chunk.choices[0].delta.reasoning_content:
#         print(chunk.choices[0].delta.reasoning_content, end='', flush=True)

#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end='', flush=True)
