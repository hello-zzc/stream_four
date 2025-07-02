import streamlit as st # 导入Streamlit库

from langchain.memory import ConversationBufferMemory # 导入对话记忆
from backend import main_logic # 导入后端

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title(" 🤖 文档对话助手")


with st.sidebar: # 在侧边栏中输入DeepSeek API密钥
    deepseek_api_key = st.text_input("请输入DeepSeek API密钥：", type="password")
    st.markdown("[获取DeepSeek API key](https://platform.deepseek.com/api_keys)")
    st.divider() # 添加分隔线
    if "chat_history" in st.session_state:
        with st.expander("历史消息"):
            for i in range(0, len(st.session_state["chat_history"]), 2):
                human_message = st.session_state["chat_history"][i] # 获取人类消息
                ai_message = st.session_state["chat_history"][i+1] # 获取AI消息
                st.write(human_message.content) # 显示人类消息内容
                st.write("AI：",ai_message.content) # 显示AI消息内容
                if i < len(st.session_state["chat_history"]) - 2:   # 如果不是最后一轮消息，则添加分隔线
                    st.divider()

if "memory" not in st.session_state: # 如果会话状态中没有记忆，则初始化记忆
    st.session_state["memory"] = ConversationBufferMemory(  # 对话记忆
        return_messages=True,# 返回消息
        memory_key="chat_history", # 记忆键
        output_key="answer" # 输出键
    )
    st.session_state["messages"] = [{"role": "ai",
                                    "content": "**铁子，我是本地文档对话客服，可以根据你上传的文档进行对话，有什么可帮你的吗？**"}] # 初始消息



pdfs = st.file_uploader("上传PDF文件：", type="pdf") # 上传PDF文件，accept_multiple_files=True允许多文件上传
for message in st.session_state["messages"]: # 遍历会话状态中的消息
    st.chat_message(message["role"]).write(message["content"]) # chat_message用于显示不同角色的消息,这里是AI角色


question = st.chat_input("你好呀",disabled=not pdfs) #disabled参数根据是否上传文件来决定是否可以输入


if question:
    if pdfs and question and not deepseek_api_key: # 如果上传了文件和问题，但没有输入API密钥
        st.info("请输入你的DeepSeek API密钥")
        st.stop()
    st.session_state["messages"].append({"role": "human", "content": question})
    st.chat_message("human").write(question)
    with st.spinner("AI正在思考中，请稍等..."):
        response = main_logic(deepseek_api_key, st.session_state["memory"],
                               pdfs, question)
        msg = {"role": "ai", "content": response["answer"]} # 保存AI角色的消息
        st.session_state["messages"].append(msg) # 将AI角色的消息添加到会话状态中
        st.chat_message("ai").write(response["answer"]) # 显示AI角色的消息
        st.session_state["chat_history"] = response["chat_history"] # 将对话历史存储到会话状态中
        st.rerun()

refresh_button = st.button("重置对话")
if refresh_button: # 如果点击了重置对话按钮
    if not deepseek_api_key: # 如果没有输入API密钥
        st.info("请输入你的DeepSeek API Key")
        st.stop()
    st.session_state["messages"] = [{"role": "ai",
                                    "content": "**铁子，我是本地文档对话客服，可以根据你上传的文档进行对话，有什么可帮你的吗？**"}] # 重置消息
    st.session_state["chat_history"] = [] # 清空对话历史
    st.session_state["memory"] = ConversationBufferMemory(  # 重置记忆
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
    st.rerun() # 重新运行应用程序



