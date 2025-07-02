from langchain.chains import ConversationalRetrievalChain  # 对话检索链
from langchain_community.document_loaders import PyPDFLoader # PDF加载库
from langchain_community.vectorstores import FAISS # 向量存储库
from langchain_community.embeddings import GPT4AllEmbeddings #嵌入模型
from langchain_deepseek import ChatDeepSeek # DeepSeek聊天模型
from langchain_text_splitters import RecursiveCharacterTextSplitter #文档分割 


def main_logic(deepseek_api_key, memory, file, ask):
    model = ChatDeepSeek(model="deepseek-reasoner", api_key=deepseek_api_key)

    file_content = file.read() # 读取上传的PDF文件内容，返回二进制数据
    temp_path = "temp_path.pdf" # 临时文件路径
    with open(temp_path, "wb") as temp_file: # 将二进制数据写入临时文件,作为本地路径
        temp_file.write(file_content)
    post = PyPDFLoader(temp_path) # 传入本地路径
    pdfs = post.load() # 加载PDF文档
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1500, # 每块文档最大字符大小
        chunk_overlap=50, # 重叠部分，避免信息丢失
        separators=["\n", "。", "！", "？", "，", "、", ""] # 中文分割符
    )
    texts = text_split.split_documents(pdfs) # 分割文档为多个文本块
    hf = GPT4AllEmbeddings(
            model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
            gpt4all_kwargs={'allow_download': 'True'}
    )
    db = FAISS.from_documents(texts, hf) # 使用FAISS向量存储库存储文本块
    retriever = db.as_retriever() # 创建检索器
    conver_chain = ConversationalRetrievalChain.from_llm( # 创建对话检索链
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = conver_chain.invoke({"chat_history": memory, "question": ask}) # 调用对话检索链，参数键传入历史对话和问题
    return response # 返回键包含 "answer" 键（回答内容）和 "source_documents" 键（引用的文档）
