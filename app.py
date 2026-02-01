import streamlit as st
import spacy
import pandas as pd
from deep_translator import GoogleTranslator
import time

# --- 1. 页面全局配置 ---
st.set_page_config(
    page_title="英语长难句・精读分析系统 Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 核心功能函数 ---

# A. 加载模型 (带缓存)
@st.cache_resource
def load_nlp_model():
    try:
        # 优先尝试加载大模型 (如果用户没装，回退到小模型)
        return spacy.load("en_core_web_trf")
    except OSError:
        st.error("请先安装模型: python -m spacy download en_core_web_trf")
        return None

# B. 翻译功能 (带缓存，防止重复请求)
@st.cache_data
def translate_text(text):
    try:
        # 使用 Google 翻译源，目标语言为简体中文
        translator = GoogleTranslator(source='auto', target='zh-CN')
        return translator.translate(text)
    except Exception as e:
        return f"翻译服务暂时不可用: {str(e)}"

# C. 辅助：术语中文化
def get_pos_cn(pos_tag):
    mapping = {
        "NOUN": "名词", "VERB": "动词", "ADJ": "形容词", "ADV": "副词",
        "PRON": "代词", "DET": "限定词", "ADP": "介词", "NUM": "数词",
        "CONJ": "连词", "CCONJ": "并列连词", "SCONJ": "从属连词", "PART": "小品词", "AUX": "助动词"
    }
    return mapping.get(pos_tag, pos_tag)

def get_dep_cn(dep_tag):
    mapping = {
        "nsubj": "主语", "nsubjpass": "被动主语", "dobj": "宾语", "pobj": "介词宾语",
        "attr": "表语", "ROOT": "核心谓语", "amod": "形容词修饰", "advmod": "副词修饰",
        "prep": "介词修饰", "compound": "复合词", "aux": "助动词", "advcl": "状语从句", 
        "relcl": "定语从句", "ccomp": "宾语/补语从句", "mark": "标记词", "det": "限定词"
    }
    return mapping.get(dep_tag, dep_tag)

# D. 核心分析逻辑
def analyze_sentence(nlp, text):
    doc = nlp(text)
    
    # 1. 提取骨架
    try:
        root = [token for token in doc if token.head == token][0]
        subjects = [w.text for w in root.lefts if w.dep_ in ["nsubj", "nsubjpass"]]
        subject = subjects[0] if subjects else "(无)"
        objects = [w.text for w in root.rights if w.dep_ in ["dobj", "attr", "acomp"]]
        obj = objects[0] if objects else "(无)"
        skeleton = {"主语": subject, "谓语": root.text, "宾语/表语": obj}
    except:
        skeleton = {"主语": "?", "谓语": "?", "宾语/表语": "?"}

    # 2. 颜色配置
    colors = {
        "ROOT": "#FFD1DC", "advcl": "#C1E1C1", "relcl": "#AEC6CF", 
        "ccomp": "#FDFD96", "default": "#F0F2F6"
    }
    token_colors = [colors["default"]] * len(doc)
    
    # 染色逻辑
    for token in doc:
        if token.head == root or token == root: token_colors[token.i] = colors["ROOT"]
    for token in doc:
        if token.dep_ in ["advcl", "relcl", "ccomp"]:
            for idx in [t.i for t in token.subtree]: token_colors[idx] = colors[token.dep_]

    # 3. 生成 HTML
    html_str = '<div style="line-height: 2.6; font-size: 20px; font-family: sans-serif;">'
    current_color, buffer_text = None, ""
    
    # 4. 生成数据表数据
    table_data = []

    for i, token in enumerate(doc):
        # 收集表格数据
        table_data.append({
            "单词": token.text,
            "原型": token.lemma_,
            "词性": get_pos_cn(token.pos_),
            "语法成分": get_dep_cn(token.dep_),
            "依赖对象": token.head.text
        })

        # 生成 HTML
        color = token_colors[i]
        tooltip = f"词性: {get_pos_cn(token.pos_)}&#10;成分: {get_dep_cn(token.dep_)}"
        is_bold = token.dep_ == "ROOT" or token.text in [skeleton["主语"], skeleton["宾语/表语"]]
        
        style = f"cursor: help; border-bottom: 2px solid {color if color != '#F0F2F6' else '#ddd'};"
        span = f'<span title="{tooltip}" style="{style}">{token.text}</span>'
        if is_bold: span = f"<b>{span}</b>"
        
        word_html = span + token.whitespace_
        
        if color != current_color:
            if buffer_text: html_str += f'<span style="background-color: {current_color}; padding: 4px 0;">{buffer_text}</span>'
            current_color, buffer_text = color, word_html
        else:
            buffer_text += word_html
            
    if buffer_text: html_str += f'<span style="background-color: {current_color}; padding: 4px 0;">{buffer_text}</span>'
    html_str += '</div>'
    
    return html_str, skeleton, pd.DataFrame(table_data)

# --- 3. 界面 UI ---

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置与图例")
    st.markdown("### 图例")
    st.markdown(f'<span style="background-color: #FFD1DC; padding:2px; border-radius:4px;">■ 主句 / 核心骨架</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="background-color: #C1E1C1; padding:2px; border-radius:4px;">■ 状语从句 (原因/条件)</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="background-color: #AEC6CF; padding:2px; border-radius:4px;">■ 定语从句 (修饰名词)</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="background-color: #FDFD96; padding:2px; border-radius:4px;">■ 宾语 / 补语从句</span>', unsafe_allow_html=True)
    
    st.divider()
    st.caption("Designed for English Learners")

# 主界面
st.title("🎓 英语长难句・精读分析系统 Pro")
st.markdown("基于 **NLP 句法树** 与 **Google 翻译** 的一站式精读工具。")

# 初始化模型
nlp = load_nlp_model()

# 输入区
default_text = "Curbs on business-method claims would be a dramatic about-face, because it was the Federal Circuit itself that introduced such patents with its 1998 decision."
text_input = st.text_area("请输入英语长难句：", value=default_text, height=120)

# 按钮行
col_btn, col_blank = st.columns([1, 5])
with col_btn:
    analyze_btn = st.button("🚀 开始深度分析", type="primary", use_container_width=True)

# 业务逻辑
if analyze_btn and text_input and nlp:
    with st.spinner("正在解析句法结构、调用翻译接口..."):
        # 1. 语法分析
        html_viz, skeleton, df_data = analyze_sentence(nlp, text_input)
        # 2. 机器翻译
        trans_result = translate_text(text_input)
        
    # --- 结果展示区 (Tab 视图) ---
    tab1, tab2, tab3 = st.tabs(["🎨 结构可视化", "📊 语法数据表", "📝 翻译与骨架"])
    
    with tab1:
        st.subheader("句法结构可视化")
        st.markdown(html_viz, unsafe_allow_html=True)
        st.info("💡 提示：鼠标悬停在单词上，可查看具体词性。")

    with tab2:
        st.subheader("单词成分详情")
        # 展示数据表
        st.dataframe(df_data, use_container_width=True)
        # 导出按钮
        csv = df_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="💾 下载分析结果 (Excel/CSV)",
            data=csv,
            file_name='grammar_analysis.csv',
            mime='text/csv',
        )

    with tab3:
        st.subheader("核心骨架")
        c1, c2, c3 = st.columns(3)
        c1.metric("主语", skeleton["主语"])
        c2.metric("谓语", skeleton["谓语"])
        c3.metric("宾语", skeleton["宾语/表语"])
        
        st.divider()
        st.subheader("参考译文")
        st.success(trans_result)

elif analyze_btn and not text_input:
    st.warning("请输入内容！")
