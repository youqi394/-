# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime
import os

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
if 'power_prediction_table' not in st.session_state:
    st.session_state.power_prediction_table = None
if 'full_debug_info' not in st.session_state:
    st.session_state.full_debug_info = ""

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
                    "weather": day["dayweather"].strip(),
                    "is_rain": 1 if "雨" in day["dayweather"] else 0
                }
                return weather_info, None
        return None, "无天气数据"
    except:
        return {"date": date, "temp_max": 25, "temp_min": 18, "weather": "晴", "is_rain": 0}, None

# -------------------------- ✅ 终极排查：全流程调试 --------------------------
@st.cache_resource(show_spinner=False)
def load_power_data_debug():
    """带完整调试信息的CSV加载函数"""
    debug_info = []
    debug_info.append("="*50)
    debug_info.append("📂 文件加载调试信息")
    debug_info.append("="*50)
    
    # 检查当前目录结构
    debug_info.append(f"📌 当前工作目录：{os.getcwd()}")
    debug_info.append(f"📌 data文件夹是否存在：{os.path.exists('data')}")
    if os.path.exists('data'):
        debug_info.append(f"📌 data文件夹里的文件：{os.listdir('data')}")
    
    file_path = "data/电量消耗.csv"
    debug_info.append(f"\n📌 尝试读取文件：{file_path}")
    debug_info.append(f"📌 文件是否存在：{os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        debug_info.append("❌ 文件不存在！")
        return None, "\n".join(debug_info)
    
    # 尝试不同编码读取
    try:
        power_df = pd.read_csv(file_path, dtype=str, encoding='utf-8')
        debug_info.append("✅ 用UTF-8编码读取成功")
    except:
        try:
            power_df = pd.read_csv(file_path, dtype=str, encoding='gbk')
            debug_info.append("✅ 用GBK编码读取成功")
        except Exception as e:
            debug_info.append(f"❌ 读取失败：{str(e)}")
            return None, "\n".join(debug_info)
    
    # 清洗数据
    power_df.columns = power_df.columns.str.strip()
    power_df = power_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    debug_info.append(f"\n📌 读取到的行数：{len(power_df)}")
    debug_info.append(f"📌 读取到的列名：{list(power_df.columns)}")
    debug_info.append(f"\n📌 CSV里的所有天气值：{power_df['天气'].unique().tolist()}")
    debug_info.append(f"\n📌 CSV前5行内容：")
    debug_info.append(power_df.head().to_string())
    
    return power_df, "\n".join(debug_info)

def statistical_prediction_debug(weather_info):
    """带完整调试信息的预测函数"""
    current_weather = weather_info['weather']
    power_df, load_debug = load_power_data_debug()
    
    debug_info = []
    debug_info.append(load_debug)
    debug_info.append("\n" + "="*50)
    debug_info.append("🔍 匹配过程调试信息")
    debug_info.append("="*50)
    debug_info.append(f"📌 当日天气（高德返回）：「{current_weather}」")
    
    if power_df is None:
        debug_info.append("❌ 无法读取CSV文件，使用默认值")
        st.session_state.full_debug_info = "\n".join(debug_info)
        return pd.DataFrame({
            "时段": ["早高峰", "晚高峰", "平峰", "低峰"],
            "天气": [current_weather]*4,
            "电量消耗": ["23.00%"]*4
        })
    
    # 匹配天气
    matched_rows = power_df[power_df['天气'] == current_weather].copy()
    debug_info.append(f"\n📌 匹配到天气「{current_weather}」的行数：{len(matched_rows)}")
    
    if len(matched_rows) > 0:
        debug_info.append(f"\n📌 匹配到的行：")
        debug_info.append(matched_rows.to_string())
    else:
        debug_info.append("❌ 没有匹配到任何行！")
        debug_info.append(f"💡 提示：CSV里的天气是 {power_df['天气'].unique().tolist()}")
        debug_info.append(f"💡 高德返回的天气是「{current_weather}」")
        debug_info.append("💡 请检查两者是否完全一致（包括空格、标点）")
    
    # 按顺序排序
    peak_order = ["早高峰", "晚高峰", "平峰", "低峰"]
    result = []
    
    for peak in peak_order:
        match = matched_rows[matched_rows['时段'] == peak]
        if not match.empty:
            power = match.iloc[0]['电量消耗']
            result.append({
                "时段": peak,
                "天气": current_weather,
                "电量消耗": power
            })
            debug_info.append(f"\n✅ {peak} → 匹配成功：{power}")
        else:
            result.append({
                "时段": peak,
                "天气": current_weather,
                "电量消耗": "23.00%"
            })
            debug_info.append(f"\n❌ {peak} → 未匹配到，使用默认值23.00%")
    
    st.session_state.full_debug_info = "\n".join(debug_info)
    return pd.DataFrame(result)

# -------------------------- 客流预测（保留原逻辑） --------------------------
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

# -------------------------- 优化求解（保留原逻辑） --------------------------
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
            st.success(f"✅ 天气：{weather_info['weather']} {weather_info['temp_min']}~{weather_info['temp_max']}℃")
            st.session_state.progress = 30
            st.session_state.current_stage = "天气已加载"

    with btn3:
        if st.button("运行统计预测"):
            if not st.session_state.weather_data:
                st.warning("⚠️ 请先读取天气")
            else:
                st.info("🔄 统计预测中...")
                is_workday = 1 if timetable_type == "工作日" else 0
                # 1. 客流预测
                hours, preds = predict_passenger_flow(dispatch_date, line, is_workday, st.session_state.weather_data)
                # 2. 电量消耗预测（终极排查版）
                power_table = statistical_prediction_debug(st.session_state.weather_data)
                
                st.session_state.predictions = preds
                st.session_state.prediction_hours = hours
                st.session_state.power_prediction_table = power_table
                
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
    st.subheader("电量消耗数据状态")
    
    power_df, debug_info = load_power_data_debug()
    
    if power_df is not None:
        st.success("✅ 成功加载 data/电量消耗.csv")
        st.dataframe(power_df, use_container_width=True)
    else:
        st.error("❌ 未找到 data/电量消耗.csv")
        st.text_area("文件加载调试信息", debug_info, height=400)

# -------------------------- 统计预测结果（✅ 带完整调试信息） --------------------------
elif page == "📊 统计预测结果":
    st.header("📊 电量消耗统计预测结果", divider="blue")
    
    if st.session_state.power_prediction_table is None:
        st.info("请先在「今日调度」页面点击「运行统计预测」")
    else:
        st.subheader(f"当日天气：{st.session_state.weather_data['weather']}")
        # 显示最终结果
        st.dataframe(st.session_state.power_prediction_table, use_container_width=True, height=200)
        
        # 下载按钮
        csv_data = st.session_state.power_prediction_table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载电量消耗预测表",
            data=csv_data,
            file_name=f"电量消耗预测_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # 显示完整调试信息
        st.divider()
        st.subheader("🔍 完整调试信息（如果结果不对，请把下面的内容发给我）")
        st.text_area("调试信息", st.session_state.full_debug_info, height=600)

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
