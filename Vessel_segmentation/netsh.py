import os
import random
import datetime
import streamlit as st
# from streamlit.server.server import Server
# from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx
from streamlit_modal import Modal as ml
from train import train_and_test
from ADUnet import ADUNet
import torch


def main():
    st.set_page_config(page_title="樱花樱花想见你", page_icon=":rainbow:", layout="wide", initial_sidebar_state="auto")
    st.title('桜——用于模型训练:heart:')
    st.sidebar.title("用于调参:balloon:")
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    ###信息框
    my_modal_login = ml(title='', key="modal_key", max_width=300)

    ###镜像源
    st.markdown("### 镜像源")
    st.write("(清华源)https://pypi.tuna.tsinghua.edu.cn/simple")
    ### Time
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    else:
        st.session_state.first_visit = False
    # 初始化全局配置
    if st.session_state.first_visit:
        # 在这里可以定义任意多个全局变量，方便程序进行调用
        st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(
            hours=8)  # Streamlit Cloud的时区是UTC，加8小时即北京时间
        st.session_state.my_random = MyRandom(random.randint(1, 1000000))
        # st.session_state.random_city_index=random.choice(range(len(st.session_state.city_mapping)))
        st.balloons()
        st.snow()
    ###训练参数和模型
    # size_ = {"小尺寸": 48, "中尺寸": 128, "大尺寸": 256}
    op_ = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    model_ = {"ADUNet": ADUNet}
    ptrain = {"载入预训练": True, "不载入预训练": False}
    epoch = {"小次数(用于判断是否成功)": 1, "中下次数": 20, "中上训练": 100, "大型训练": 500,
             "自定义训练次数": "epochs"}

    ##------------------------------------------
    # size = st.sidebar.selectbox("选择图片尺寸", size_.keys())
    # st.sidebar.write("尺寸大小:", size_[size])
    size=st.sidebar.text_input("输入图片尺寸",value=48)

    op = st.sidebar.selectbox("选择一个优化器", op_.keys())
    model = st.sidebar.selectbox("选择一个模型", model_.keys())
    pre_train = st.sidebar.selectbox("是否载入预训练", ptrain.keys())
    ep = st.sidebar.selectbox("选择训练次数", epoch)
    if ep == "自定义训练次数":
        epochs = st.sidebar.text_input("请输入训练次数")
        st.sidebar.write("num:", epochs)
        ep = epochs
    elif ep != "自定义训练次数":
        ep = epoch[ep]
        st.sidebar.write("训练次数:", ep)
    train_ = True
    # if st.sidebar.button("停止", help="用于停止训练"):
    #     train_ = False
    ###进行训练
    st.markdown("### 训练处")
    if train_ and st.sidebar.button("开始训练", help="用于开始模型的训练"):
        train(op_[op], model_[model], size, ep, ptrain[pre_train])


    # The END
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)



class MyRandom:
    def __init__(self, num):
        self.random_num = num


def train(op, model, size, ep, ptrain):
    train__ = train_and_test(op, model, size, ep, ptrain)
    return train__


if __name__ == '__main__':
    main()
