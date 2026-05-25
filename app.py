# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime

# -------------------------- 全局配置 --------------------------
st.set_page_config(
    page_title="智能公交调度系统",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stButton>button {
    height: 50px;
    font-size: 16px;
    width: 100%;
    border-radius: 12px;
    border: none;
    background-color: #1f77b4;
    color: white;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #155a8a;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
}
.stMetric {
    background-color: #f8f9fa;
    padding: 12px;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
}
.stProgress > div > div {
    background-color: #1f77b4;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #2c3e50;
    font-weight: 600;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------------- 初始化会话状态 --------------------------
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = "等待开始"
if 'timetable_data' not in st.session_state:
    st.session_state.timetable_data = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'prediction_hours' not in st.session_state:
    st.session_state.prediction_hours = None
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None
if 'schedule_data' not in st.session_state:
    st.session_state.schedule_data = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'current_gap' not in st.session_state:
    st.session_state.current_gap = 0.85
if 'current_objective' not in st.session_state:
    st.session_state.current_objective = 50.0
if 'convergence_data' not in st.session_state:
    st.session_state.convergence_data = []
if 'solve_log' not in st.session_state:
    st.session_state.solve_log = []
if 'power_consumption_data' not in st.session_state:
    st.session_state.power_consumption_data = None
# if 'runtime_data' not in st.session_state:
#     st.session_state.runtime_data = None
if 'power_prediction' not in st.session_state:
    st.session_state.power_prediction = None
# if 'runtime_prediction' not in st.session_state:
#     st.session_state.runtime_prediction = None

# -------------------------- 工具函数 --------------------------
def add_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.solve_log.append(f"[INFO] {timestamp} - {message}")

# -------------------------- 天气获取 --------------------------
def get_weather_forecast(date):
    WEATHER_API_KEY = "e088a35c897818780a479973d4623063"
    try:
        city_code = "110000"
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_code}&key={WEATHER_API_KEY}&extensions=all"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("status") != "1":
            return None, "API错误"
        target = date.strftime("%Y-%m-%d")
        for day in data["forecasts"][0]["casts"]:
            if day["date"] == target:
                weather_info = {
                    "date": date,
                    "temp_max": int(day["daytemp"]),
                    "temp_min": int(day["nighttemp"]),
                    "weather": day["dayweather"],
                    "is_rain": 1 if "雨" in day["dayweather"] else 0
                }
                return weather_info, None
        return None, "无天气数据"
    except:
        return {"date": date, "temp_max": 25, "temp_min": 18, "weather": "晴", "is_rain": 0}, None

# -------------------------- 统计预测：只启用电量，运行时间已注释 --------------------------
@st.cache_resource
def load_statistical_data():
    power_data = None
    # runtime_data = None
    try:
        power_data = pd.read_csv("data/电量消耗.csv")
        add_log("✅ 成功加载 data/电量消耗.csv")
    except Exception as e:
        add_log(f"⚠️ 未找到 data/电量消耗.csv")

    # try:
    #     runtime_data = pd.read_csv("data/运行时间.csv")
    #     add_log("✅ 成功加载 data/运行时间.csv")
    # except Exception as e:
    #     add_log(f"⚠️ 未找到 data/运行时间.csv")

    return power_data  # , runtime_data

def statistical_prediction(weather_info):
    hours = list(range(6, 22))
    current_weather = weather_info['weather']
    power_df = load_statistical_data()  # , runtime_df

    # 电量消耗预测（正常运行）
    power_pred = []
    if power_df is not None:
        for hour in hours:
            time_str = f"{hour}:00"
            match_row = power_df[(power_df['时段'] == time_str) & (power_df['天气'] == current_weather)]
            if not match_row.empty:
                power_pred.append(float(match_row.iloc[0]['电量消耗']))
            else:
                power_pred.append(10.0)
    else:
        power_pred = [10.0] * len(hours)

    # 运行时间预测（已全部注释，不运行）
    # runtime_pred = []
    # if runtime_df is not None:
    #     for hour in hours:
    #         time_str = f"{hour}:00"
    #         match_row = runtime_df[(runtime_df['时段'] == time_str) & (runtime_df['天气'] == current_weather)]
    #         if not match_row.empty:
    #             runtime_pred.append(float(match_row.iloc[0]['运行时间']))
    #         else:
    #             runtime_pred.append(45.0)
    # else:
    #     runtime_pred = [45.0] * len(hours)

    return power_pred  # , runtime_pred

# -------------------------- 客流预测 --------------------------
def predict_passenger_flow(date, line_id, is_workday, weather_data):
    hours = list(range(6, 22))
    base_flow = 150 if is_workday else 100
    rain_factor = 0.8 if weather_data and weather_data['is_rain'] else 1.0
    predictions = []
    for hour in hours:
        if 7 <= hour <= 9:
            flow = base_flow * 2.5 * rain_factor
        elif 17 <= hour <= 19:
            flow = base_flow * 2.2 * rain_factor
        elif 6 <= hour <= 21:
            flow = base_flow * rain_factor
        else:
            flow = base_flow * 0.5 * rain_factor
        predictions.append(round(flow * (0.9 + np.random.random() * 0.2)))
    return hours, predictions

# -------------------------- 优化求解 --------------------------
def optimize_schedule(predictions, vehicle_count, initial_battery, solve_time_limit):
    add_log("开始初始化优化模型")
    st.session_state.convergence_data = []
    hours = list(range(6, 22))
    n_hours = len(hours)
    n_vehicles = vehicle_count
    model = gp.Model("bus_scheduling")
    model.setParam('TimeLimit', solve_time_limit)
    model.setParam('OutputFlag', 1)

    def callback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            if obj < 1e100:
                st.session_state.current_objective = obj
                st.session_state.current_gap = (obj - bound) / obj if obj != 0 else 0
                st.session_state.convergence_data.append((len(st.session_state.convergence_data)+1, obj))

    x = model.addVars(n_vehicles, n_hours, vtype=GRB.BINARY, name="x")
    y = model.addVars(n_hours, vtype=GRB.CONTINUOUS, name="y")
    obj = 0
    for j in range(n_hours):
        obj += predictions[j] * y[j] * 0.1
        for i in range(n_vehicles):
            obj += x[i, j] * 50
    model.setObjective(obj, GRB.MINIMIZE)

    for j in range(n_hours):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n_vehicles)) >= 1)
    for j in range(n_hours):
        departures = gp.quicksum(x[i, j] for i in range(n_vehicles))
        model.addConstr(y[j] == 60 / departures)
        model.addConstr(y[j] <= 15)
    for i in range(n_vehicles):
        for j in range(n_hours - 4):
            model.addConstr(gp.quicksum(x[i, j+k] for k in range(5)) <= 4)
    for i in range(n_vehicles):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n_hours)) * 10 <= initial_battery)

    model.optimize(callback)
    add_log(f"求解完成，最优目标值：{model.ObjVal:.2f}")
    schedule = []
    for j in range(n_hours):
        hour = hours[j]
        departures = [f"车{i+1:02d}" for i in range(n_vehicles) if x[i, j].X > 0.5]
        interval = 60 / len(departures) if departures else 60
        for k, vehicle in enumerate(departures):
            minute = round(k * interval)
            depart_time = f"{hour:02d}:{minute:02d}"
            arrive_time = f"{hour:02d}:{minute+45:02d}" if minute+45 < 60 else f"{hour+1:02d}:{minute-15:02d}"
            schedule.append({
                "车辆编号": vehicle, "发车时间": depart_time, "到达时间": arrive_time,
                "司机": f"司机{ord(vehicle[-2:])%20+1:02d}", "电量消耗": "10%"
            })
    return model, pd.DataFrame(schedule)

# -------------------------- 侧边栏 --------------------------
st.sidebar.title("🚌 智能公交调度系统")
st.sidebar.markdown("""<style>[data-testid="stSidebar"] {background-color: #f0f5fa;}</style>""", unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("功能模块", ["📅 今日调度", "📊 数据管理", "📊 统计预测结果", "⚙️ 优化求解", "📋 排班结果"])
st.sidebar.divider()
st.sidebar.info("智能公交调度系统")

# -------------------------- 今日调度 --------------------------
if page == "📅 今日调度":
    st.header("🚌 智能公交调度", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        dispatch_date = st.date_input("调度日期", datetime.now().date())
        line = st.selectbox("线路/场站", ["1路", "2路", "3路", "4路", "5路"])
        timetable_type = st.selectbox("班次表", ["工作日", "周末", "节假日"])
    with col2:
        vehicle_count = st.number_input("当日车辆数", 1, 50, 15)
        initial_battery = st.number_input("初始电量（%）", 0, 100, 100)
        solve_time = st.number_input("求解时间上限（秒）", 60, 3600, 300)
    st.divider()

    btn1, btn2, btn3, btn4, btn5 = st.columns(5, gap="small")
    with btn1:
        if st.button("读取班次表"):
            st.session_state.start_time = time.time()
            try:
                st.session_state.timetable_data = pd.read_csv("data/weekday.csv")
                st.success("✅ 班次表读取成功")
            except:
                st.session_state.timetable_data = pd.DataFrame({
                    "线路编号":["1路"]*10,"发车时间":[f"{6+i//2:02d}:{i%2*30:02d}"for i in range(10)],"车辆编号":[f"车{i%5+1:02d}"for i in range(10)]
                })
                st.warning("⚠️ 使用示例班次数据")
            st.session_state.progress = 24
            st.session_state.current_stage = "班次已加载"

    with btn2:
        if st.button("读取天气"):
            weather_info, err = get_weather_forecast(dispatch_date)
            st.session_state.weather_data = weather_info
            st.success(f"✅ 天气：{weather_info['weather']}")
            st.session_state.progress = 30
            st.session_state.current_stage = "天气已加载"

    with btn3:
        if st.button("运行统计预测"):
            if not st.session_state.weather_data:
                st.warning("⚠️ 请先读取天气")
            else:
                st.info("🔄 统计预测中...")
                is_workday = 1 if timetable_type == "工作日" else 0
                hours, preds = predict_passenger_flow(dispatch_date, line, is_workday, st.session_state.weather_data)
                power_pred = statistical_prediction(st.session_state.weather_data)
                
                st.session_state.predictions = preds
                st.session_state.prediction_hours = hours
                st.session_state.power_prediction = power_pred
                # st.session_state.runtime_prediction = runtime_pred
                
                st.session_state.progress = 60
                st.session_state.current_stage = "统计预测完成"
                st.success("✅ 统计预测完成！")

    with btn4:
        if st.button("开始优化求解"):
            if not st.session_state.predictions:
                st.warning("⚠️ 请先运行统计预测")
            else:
                st.info("🔄 求解中...")
                model, df = optimize_schedule(st.session_state.predictions, vehicle_count, initial_battery, solve_time)
                st.session_state.optimization_result = model
                st.session_state.schedule_data = df
                st.session_state.progress = 90
                st.session_state.current_stage = "优化求解完成"
                st.success("✅ 优化求解完成！")

    with btn5:
        if st.button("导出排班结果"):
            if st.session_state.schedule_data is not None:
                csv = st.session_state.schedule_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 下载排班表", csv, f"公交排班表_{dispatch_date.strftime('%Y%m%d')}.csv")
                st.session_state.progress = 100
                st.session_state.current_stage = "全部完成"
            else:
                st.warning("⚠️ 请先完成求解")

    st.divider()
    st.progress(st.session_state.progress, text=f"进度 {st.session_state.progress}%")
    st.divider()
    s1,s2,s3,s4,s5 = st.columns(5,gap="small")
    s1.metric("当前阶段", st.session_state.current_stage)
    s2.metric("已用时间", f"{int(time.time()-st.session_state.start_time)}s" if st.session_state.start_time else "0s")
    s3.metric("预计剩余", f"{int((100-st.session_state.progress)*0.5)}s" if st.session_state.progress<100 else "0s")
    s4.metric("Gap", f"{st.session_state.current_gap:.2f}")
    s5.metric("目标值", f"{st.session_state.current_objective:.2f}")

# -------------------------- 数据管理 --------------------------
elif page == "📊 数据管理":
    st.header("📊 数据管理", divider="blue")
    st.subheader("统计数据状态（从 data 文件夹读取）")
    p = load_statistical_data()  # , r
    c1,c2 = st.columns(2)
    c1.success("✅ 电量消耗.csv 已加载") if p is not None else c1.error("❌ 未找到 data/电量消耗.csv")
    # c2.success("✅ 运行时间.csv 已加载") if r is not None else c2.error("❌ 未找到 data/运行时间.csv")

# -------------------------- 统计预测结果 --------------------------
elif page == "📊 统计预测结果":
    st.header("📊 统计预测结果", divider="blue")
    if st.session_state.predictions is None:
        st.info("请先运行统计预测")
    else:
        t = [f"{h}:00" for h in st.session_state.prediction_hours]
        st.subheader("客流预测")
        st.line_chart(pd.DataFrame({"时间":t,"客流":st.session_state.predictions}),x="时间",y="客流")
        
        st.subheader("电量消耗预测 %")
        st.line_chart(pd.DataFrame({"时间":t,"电量消耗":st.session_state.power_prediction}),x="时间",y="电量消耗")
        
        # st.subheader("运行时间预测 分钟")
        # st.line_chart(pd.DataFrame({"时间":t,"运行时间":st.session_state.runtime_prediction}),x="时间",y="运行时间")

# -------------------------- 优化求解 --------------------------
elif page == "⚙️ 优化求解":
    st.header("⚙️ 优化求解", divider="blue")
    if st.session_state.optimization_result:
        st.metric("最优目标值", f"{st.session_state.optimization_result.ObjVal:.2f}")

# -------------------------- 排班结果 --------------------------
elif page == "📋 排班结果":
    st.header("📋 排班结果", divider="blue")
    if st.session_state.schedule_data is not None:
        st.dataframe(st.session_state.schedule_data, use_container_width=True)
    else:
        st.info("请先完成优化求解")
