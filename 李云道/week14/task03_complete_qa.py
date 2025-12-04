'''
结合rag tool筛选 加 mcp tool执行完成 整个问答。
  - 步骤2: 定义mcp服务，并且将每一个公式整理为mcp 的一个可执行的tool 或 sympy的计算过程。
  - 步骤3: 得到用户提问的时候，需要选择对应的tool
    - 通过rag的步骤（用户的提问 与 公式 进行相似度计算，也可以加入rerank 过程，筛选top1/3公式） -》 tool 白名单
  - 步骤4: 调用对应的tool，执行，得到结果，汇总得到回答。
'''


def qa(query: str) -> str:
    messages = f"user:{query}\nassistant:Nice"
    return messages


if __name__ == '__main__':
    query = "学生张三的情况如下：学习时常25小时，出勤率95%，平时测验平均分65%，课堂参与度4分。他的综合表现分数是多少"
