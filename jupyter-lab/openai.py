import openai
import time

openai.api_key = "sk-xxxx"
system_prompt = "你是一个专业的Java开发人员"

def ask_question(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content
    return answer

# 初始化对话
print("ChatGPT: 你好！我是一个可以回答问题的AI。如果你想结束对话，请输入 '退出'。")

while True:
    user_input = input("User: ")

    if user_input == "退出":
        print("ChatGPT: 再见！")
        break

    # 提问并获取回答，计时开始
    start_time = time.time()
    answer = ask_question(user_input)
    end_time = time.time()
    execution_time = end_time - start_time
    # 计时结束

    # 输出回答和耗时
    print("ChatGPT:", answer)
    print("耗时: {:.2f}s".format(execution_time))