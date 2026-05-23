# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import time

# -------------------------- 全局配置（必须放在最开头） --------------------------
st.set_page_config(
    page_title="智能公交调度系统",
    page_icon="🚌",
    layout="wide",  # 宽屏模式，铺满浏览器
    initial_sidebar_state="expanded"
)

# 隐藏Streamlit默认的右上角菜单和底部水印（可选）
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stButton>button {
    height: 50px;
    font-size: 16px;
    width: 100%;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------------- 初始化会话状态（保存进度和数据） --------------------------
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = "等待开始"
if 'timetable_data' not in st.session_state:
    st.session_state.timetable_data = None

# -------------------------- 侧边栏导航（对应你左侧的5个按钮） --------------------------
st.sidebar.title("🚍 智能公交调度系统")
st.sidebar.divider()
page = st.sidebar.radio(
    "功能模块",
    ["📅 今日调度", "📊 数据管理", "🤖 AI预测结果", "⚙️ 优化求解", "📋 排班结果"]
)
st.sidebar.divider()
st.sidebar.info("AI预测-优化建模 智能公交调度")

# -------------------------- 页面1：今日调度（和你Qt界面完全一致） --------------------------
if page == "📅 今日调度":
    st.header("AI预测-优化建模 智能公交调度")
    st.divider()

    # 调度参数设置（左右两列布局）
    col1, col2 = st.columns(2)

    with col1:
        dispatch_date = st.date_input("调度日期")
        line = st.selectbox("线路/场站", ["1路", "2路", "3路", "4路", "5路"])
        timetable_type = st.selectbox("班次表", ["工作日", "周末", "节假日"])

    with col2:
        vehicle_count = st.number_input("当日车辆数", min_value=1, max_value=50, value=1)
        initial_battery = st.number_input("初始电量（%）", min_value=0, max_value=100, value=100)
        confidence = st.selectbox("预测置信水平", ["75%", "80%", "85%", "90%", "95%"])
        solve_time = st.number_input("求解时间上限（秒）", min_value=60, max_value=3600, value=300)

    st.divider()

    # 功能按钮（5个按钮一行）
    btn1, btn2, btn3, btn4, btn5 = st.columns(5)

    with btn1:
        if st.button("读取班次表"):
            # 示例：读取CSV文件（后续替换成你的真实文件路径）
            try:
                st.session_state.timetable_data = pd.read_csv("data/weekday.csv")
                st.success("✅ 班次表读取成功！")
                st.session_state.current_stage = "数据加载完成"
                st.session_state.progress = 24
            except:
                # 如果没有CSV文件，显示示例数据
                st.session_state.timetable_data = pd.DataFrame({
                    "线路编号": ["1路", "1路", "1路", "2路", "2路"],
                    "发车时间": ["06:00", "06:10", "06:20", "06:05", "06:15"],
                    "车辆编号": ["车01", "车02", "车03", "车04", "车05"]
                })
                st.warning("⚠️ 未找到数据文件，已加载示例数据")
                st.session_state.current_stage = "示例数据加载完成"
                st.session_state.progress = 24

    with btn2:
        if st.button("读取天气"):
            st.success("✅ 天气数据读取成功！")
            st.session_state.current_stage = "天气数据加载完成"
            st.session_state.progress = 30

    with btn3:
        if st.button("运行AI预测"):
            st.info("🔄 AI客流预测中...")
            # 模拟预测过程
            for i in range(30, 60):
                st.session_state.progress = i
                time.sleep(0.05)
            st.success("✅ AI预测完成！")
            st.session_state.current_stage = "AI预测完成"
            st.session_state.progress = 60

    with btn4:
        if st.button("开始优化求解"):
            st.info("🔄 优化求解中...")
            # 模拟求解过程
            for i in range(60, 90):
                st.session_state.progress = i
                time.sleep(0.05)
            st.success("✅ 优化求解完成！")
            st.session_state.current_stage = "优化求解完成"
            st.session_state.progress = 90

    with btn5:
        if st.button("导出排班结果"):
            st.success("✅ 排班结果已导出为CSV文件！")
            st.session_state.current_stage = "全部完成"
            st.session_state.progress = 100

    st.divider()

    # 进度条
    st.progress(st.session_state.progress, text=f"当前进度：{st.session_state.progress}%")

    st.divider()

    # 底部状态显示
    status1, status2, status3, status4, status5 = st.columns(5)
    with status1:
        st.metric("当前阶段", st.session_state.current_stage)
    with status2:
        st.metric("已用时间", "12秒")
    with status3:
        st.metric("预计剩余", "38秒")
    with status4:
        st.metric("当前Gap", "0.85")
    with status5:
        st.metric("当前目标值", "12.5")

# -------------------------- 页面2：数据管理 --------------------------
elif page == "📊 数据管理":
    st.header("数据管理模块")
    st.divider()

    st.subheader("班次数据管理")
    if st.session_state.timetable_data is not None:
        st.dataframe(st.session_state.timetable_data, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="下载班次表",
                data=st.session_state.timetable_data.to_csv(index=False),
                file_name="timetable.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("清空数据"):
                st.session_state.timetable_data = None
                st.rerun()
    else:
        st.info("请先在「今日调度」页面读取班次表")

    st.divider()

    st.subheader("上传新数据")
    uploaded_file = st.file_uploader("选择CSV文件", type="csv")
    if uploaded_file is not None:
        st.session_state.timetable_data = pd.read_csv(uploaded_file)
        st.success("✅ 数据上传成功！")
        st.dataframe(st.session_state.timetable_data, use_container_width=True)

# -------------------------- 页面3：AI预测结果 --------------------------
elif page == "🤖 AI预测结果":
    st.header("AI客流预测结果")
    st.divider()

    st.subheader("线路客流预测曲线")
    # 示例预测数据
    hours = list(range(6, 22))
    passengers = [120, 180, 250, 320, 280, 220, 190, 210, 240, 290, 350, 310, 260, 200, 150, 100]

    # 用Streamlit原生图表展示
    chart_data = pd.DataFrame({
        "时间": [f"{h}:00" for h in hours],
        "预测客流": passengers
    })
    st.line_chart(chart_data, x="时间", y="预测客流", use_container_width=True)

    st.divider()

    st.subheader("预测结果统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("早高峰最大客流", "350人")
    with col2:
        st.metric("晚高峰最大客流", "320人")
    with col3:
        st.metric("日均总客流", "3680人")
    with col4:
        st.metric("预测准确率", "92.5%")

# -------------------------- 页面4：优化求解 --------------------------
elif page == "⚙️ 优化求解":
    st.header("优化求解过程监控")
    st.divider()

    st.subheader("求解参数")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("求解时间上限", "300秒")
    with col2:
        st.metric("迭代次数", "1250次")
    with col3:
        st.metric("当前最优解", "12.5")

    st.divider()

    st.subheader("目标函数收敛曲线")
    # 示例收敛数据
    iterations = list(range(1, 101))
    objective = [50 - i * 0.375 for i in iterations]

    chart_data = pd.DataFrame({
        "迭代次数": iterations,
        "目标函数值": objective
    })
    st.line_chart(chart_data, x="迭代次数", y="目标函数值", use_container_width=True)

    st.divider()

    st.subheader("求解日志")
    log_text = """
[INFO] 2026-05-23 14:30:00 - 开始加载数据
[INFO] 2026-05-23 14:30:02 - 数据加载完成，共120条班次数据
[INFO] 2026-05-23 14:30:03 - 初始化优化模型
[INFO] 2026-05-23 14:30:05 - 开始迭代求解
[INFO] 2026-05-23 14:30:15 - 迭代100次，当前目标值：25.3
[INFO] 2026-05-23 14:30:25 - 迭代500次，当前目标值：15.8
[INFO] 2026-05-23 14:30:35 - 迭代1000次，当前目标值：12.7
[INFO] 2026-05-23 14:30:40 - 找到最优解，目标值：12.5
[INFO] 2026-05-23 14:30:41 - 求解完成
    """
    st.text_area("求解日志", log_text, height=300, disabled=True)

# -------------------------- 页面5：排班结果 --------------------------
elif page == "📋 排班结果":
    st.header("最终排班结果")
    st.divider()

    st.subheader("排班数据统计汇总")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总班次", "120班")
    with col2:
        st.metric("使用车辆数", "15辆")
    with col3:
        st.metric("平均发车间隔", "5分钟")
    with col4:
        st.metric("总运营时长", "16小时")

    st.divider()

    st.subheader("详细排班表")
    # 示例排班数据
    schedule_data = pd.DataFrame({
        "车辆编号": ["车01", "车01", "车01", "车02", "车02", "车02"],
        "发车时间": ["06:00", "06:30", "07:00", "06:10", "06:40", "07:10"],
        "到达时间": ["06:45", "07:15", "07:45", "06:55", "07:25", "07:55"],
        "司机": ["张师傅", "张师傅", "张师傅", "李师傅", "李师傅", "李师傅"],
        "电量消耗": ["15%", "14%", "16%", "13%", "15%", "14%"]
    })
    st.dataframe(schedule_data, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="下载完整排班表",
            data=schedule_data.to_csv(index=False),
            file_name="bus_schedule.csv",
            mime="text/csv"
        )
    with col2:
        st.button("打印排班表")