# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import joblib
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------- 全局配置（必须放在最开头） --------------------------
st.set_page_config(
    page_title="智能公交调度系统",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏Streamlit默认的右上角菜单和底部水印 + 全局美化样式【仅新增样式，无逻辑修改】
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
.stDivider {
    border-color: #e9ecef;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------------- 全局常量与配置 --------------------------
# 已使用你的高德 KEY
CITY_ID = "101010100"

# -------------------------- 初始化会话状态（保存进度和数据） --------------------------
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
# ✅ 新增：统计预测数据存储
if 'power_consumption_data' not in st.session_state:
    st.session_state.power_consumption_data = None
if 'runtime_data' not in st.session_state:
    st.session_state.runtime_data = None
if 'power_prediction' not in st.session_state:
    st.session_state.power_prediction = None
if 'runtime_prediction' not in st.session_state:
    st.session_state.runtime_prediction = None

# -------------------------- 工具函数 --------------------------
def add_log(message):
    """添加求解日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.solve_log.append(f"[INFO] {timestamp} - {message}")

# -------------------------- ✅ 仅这里修改：真实高德天气（完整版代码不动） --------------------------
def get_weather_forecast(date):
    """获取真实天气（高德）"""
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
        weather_info = {
            "date": date,
            "temp_max": 25,
            "temp_min": 18,
            "weather": "晴",
            "is_rain": 0
        }
        return weather_info, None

# -----------------------------------------------------------------------------

@st.cache_resource
def load_ai_models():
    """加载AI预测模型（只加载一次）"""
    try:
        model = joblib.load("data/flow_prediction_model.pkl")
        scaler = joblib.load("data/scaler.pkl")
        pca_flow = joblib.load("data/pca_flow.pkl")
        pca_congestion = joblib.load("data/pca_congestion.pkl")
        return model, scaler, pca_flow, pca_congestion, None
    except:
        add_log("未找到预训练模型，使用内置默认模型")
        return None, None, None, None, "未找到预训练模型，将使用默认预测数据"

def predict_passenger_flow(date, line_id, is_workday, weather_data):
    """客流预测（保留原逻辑）"""
    model, scaler, pca_flow, pca_congestion, error = load_ai_models()
    
    hours = list(range(6, 22))
    predictions = []
    
    if error:
        base_flow = 150 if is_workday else 100
        rain_factor = 0.8 if weather_data and weather_data['is_rain'] else 1.0
        
        for hour in hours:
            if 7 <= hour <= 9:
                flow = base_flow * 2.5 * rain_factor
            elif 17 <= hour <= 19:
                flow = base_flow * 2.2 * rain_factor
            elif 6 <= hour <= 21:
                flow = base_flow * rain_factor
            else:
                flow = base_flow * 0.5 * rain_factor
            flow = flow * (0.9 + np.random.random() * 0.2)
            predictions.append(round(flow))
    else:
        for hour in hours:
            features = pd.DataFrame({
                'hour': [hour],
                'day_of_week': [date.weekday()],
                'is_workday': [is_workday],
                'is_holiday': [0],
                'temp': [weather_data['temp_max'] if weather_data else 25],
                'is_rain': [weather_data['is_rain'] if weather_data else 0]
            })
            flow_features = features[['hour', 'is_workday']]
            congestion_features = features[['temp', 'is_rain']]
            flow_scaled = scaler.transform(flow_features)
            congestion_scaled = scaler.transform(congestion_features)
            features['flow_factor'] = pca_flow.transform(flow_scaled)[0][0]
            features['congestion_factor'] = pca_congestion.transform(congestion_scaled)[0][0]
            pred = model.predict(features[['hour', 'day_of_week', 'is_workday', 'flow_factor', 'congestion_factor']])[0]
            predictions.append(round(pred))
    
    return hours, predictions

# ✅ 修改：从GitHub的date文件夹读取CSV文件进行统计预测
@st.cache_resource
def load_statistical_data():
    """从date文件夹加载统计数据（只加载一次）"""
    power_data = None
    runtime_data = None
    
    try:
        # 读取电量消耗数据
        power_data = pd.read_csv("date/电量消耗.csv")
        add_log("✅ 成功加载date/电量消耗.csv")
    except Exception as e:
        add_log(f"⚠️ 未找到date/电量消耗.csv，将使用默认值：{str(e)}")
    
    try:
        # 读取运行时间数据
        runtime_data = pd.read_csv("date/运行时间.csv")
        add_log("✅ 成功加载date/运行时间.csv")
    except Exception as e:
        add_log(f"⚠️ 未找到date/运行时间.csv，将使用默认值：{str(e)}")
    
    return power_data, runtime_data

def statistical_prediction(weather_info):
    """基于date文件夹的CSV文件进行统计预测"""
    hours = list(range(6, 22))
    current_weather = weather_info['weather']
    
    # 加载统计数据（只加载一次）
    power_df, runtime_df = load_statistical_data()
    
    # 电量消耗预测
    power_pred = []
    if power_df is not None:
        for hour in hours:
            time_str = f"{hour}:00"
            # 精确匹配时段和天气
            match_row = power_df[(power_df['时段'] == time_str) & (power_df['天气'] == current_weather)]
            if not match_row.empty:
                power_pred.append(float(match_row.iloc[0]['电量消耗']))
            else:
                # 无匹配时使用默认值
                power_pred.append(10.0)
    else:
        # 未找到文件时使用默认值
        power_pred = [10.0] * len(hours)
    
    # 运行时间预测
    runtime_pred = []
    if runtime_df is not None:
        for hour in hours:
            time_str = f"{hour}:00"
            match_row = runtime_df[(runtime_df['时段'] == time_str) & (runtime_df['天气'] == current_weather)]
            if not match_row.empty:
                runtime_pred.append(float(match_row.iloc[0]['运行时间']))
            else:
                runtime_pred.append(45.0)
    else:
        runtime_pred = [45.0] * len(hours)
    
    return power_pred, runtime_pred

def optimize_schedule(predictions, vehicle_count, initial_battery, solve_time_limit):
    """Gurobi真实优化求解（原逻辑完全保留）"""
    add_log("开始初始化优化模型")
    
    st.session_state.convergence_data = []
    st.session_state.solve_log = []
    
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
    
    add_log("模型初始化完成，开始求解")
    model.optimize(callback)
    add_log(f"求解完成，最优目标值：{model.ObjVal:.2f}")
    
    schedule = []
    for j in range(n_hours):
        hour = hours[j]
        departures = []
        for i in range(n_vehicles):
            if x[i, j].X > 0.5:
                departures.append(f"车{i+1:02d}")
        interval = 60 / len(departures)
        for k, vehicle in enumerate(departures):
            minute = round(k * interval)
            depart_time = f"{hour:02d}:{minute:02d}"
            arrive_time = f"{hour:02d}:{minute+45:02d}" if minute+45 < 60 else f"{hour+1:02d}:{minute-15:02d}"
            schedule.append({
                "车辆编号": vehicle,
                "发车时间": depart_time,
                "到达时间": arrive_time,
                "司机": f"司机{ord(vehicle[-2:])%20+1:02d}",
                "电量消耗": f"{10}%"
            })
    
    schedule_df = pd.DataFrame(schedule)
    return model, schedule_df

# -------------------------- 侧边栏导航 --------------------------
st.sidebar.title("🚌 智能公交调度系统")
# 侧边栏背景美化
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #f0f5fa;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio(
    "功能模块",
    ["📅 今日调度", "📊 数据管理", "📊 统计预测结果", "⚙️ 优化求解", "📋 排班结果"]
)
st.sidebar.divider()
st.sidebar.info("AI预测-优化建模 智能公交调度")

# -------------------------- 页面1：今日调度 --------------------------
if page == "📅 今日调度":
    st.header("🚌 AI预测-优化建模 智能公交调度", divider="blue")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        dispatch_date = st.date_input("调度日期", value=datetime.now().date())
        line = st.selectbox("线路/场站", ["1路", "2路", "3路", "4路", "5路"])
        timetable_type = st.selectbox("班次表", ["工作日", "周末", "节假日"])
    with col2:
        vehicle_count = st.number_input("当日车辆数", min_value=1, max_value=50, value=15)
        initial_battery = st.number_input("初始电量（%）", min_value=0, max_value=100, value=100)
        confidence = st.selectbox("预测置信水平", ["75%", "80%", "85%", "90%", "95%"])
        solve_time = st.number_input("求解时间上限（秒）", min_value=60, max_value=3600, value=300)

    st.divider()
    btn1, btn2, btn3, btn4, btn5 = st.columns(5, gap="small")

    with btn1:
        if st.button("读取班次表"):
            st.session_state.start_time = time.time()
            try:
                st.session_state.timetable_data = pd.read_csv("data/weekday.csv")
                st.success("✅ 班次表读取成功！")
                st.session_state.current_stage = "数据加载完成"
                st.session_state.progress = 24
                add_log("班次表读取成功，共{}条记录".format(len(st.session_state.timetable_data)))
            except:
                st.session_state.timetable_data = pd.DataFrame({
                    "线路编号": ["1路"]*10,
                    "发车时间": [f"{6+i//2:02d}:{i%2*30:02d}" for i in range(10)],
                    "车辆编号": [f"车{i%5+1:02d}" for i in range(10)]
                })
                st.warning("⚠️ 未找到数据文件，已加载示例班次数据")
                st.session_state.current_stage = "示例数据加载完成"
                st.session_state.progress = 24
                add_log("未找到数据文件，加载示例班次数据")

    with btn2:
        if st.button("读取天气"):
            with st.spinner("正在获取天气预报..."):
                weather_info, error = get_weather_forecast(dispatch_date)
                if error:
                    st.error(f"❌ {error}")
                    st.info("将使用默认天气数据继续")
                else:
                    st.session_state.weather_data = weather_info
                    st.success(f"✅ 天气数据读取成功！{weather_info['weather']}，{weather_info['temp_min']}~{weather_info['temp_max']}℃")
                    add_log(f"天气数据读取成功：{weather_info['weather']}，{weather_info['temp_min']}~{weather_info['temp_max']}℃")
                
                st.session_state.current_stage = "天气数据加载完成"
                st.session_state.progress = 30

    with btn3:
        if st.button("运行统计预测"):
            if st.session_state.weather_data is None:
                st.warning("⚠️ 请先读取天气数据")
            else:
                st.info("🔄 统计预测中...")
                progress_bar = st.progress(0)
                
                is_workday = 1 if timetable_type == "工作日" else 0
                # 1. 保留原客流预测
                hours, predictions = predict_passenger_flow(
                    dispatch_date, line, is_workday, st.session_state.weather_data
                )
                # 2. 统计预测电量消耗和运行时间（从date文件夹读取）
                power_pred, runtime_pred = statistical_prediction(st.session_state.weather_data)
                
                # 更新进度条
                for i in range(30, 60):
                    progress_bar.progress(i/100)
                    time.sleep(0.02)
                
                # 保存所有预测结果
                st.session_state.predictions = predictions
                st.session_state.prediction_hours = hours
                st.session_state.power_prediction = power_pred
                st.session_state.runtime_prediction = runtime_pred
                
                st.success("✅ 统计预测完成！")
                st.session_state.current_stage = "统计预测完成"
                st.session_state.progress = 60
                add_log(f"统计预测完成，共预测{len(hours)}个时段")

    with btn4:
        if st.button("开始优化求解"):
            if st.session_state.predictions is None:
                st.warning("⚠️ 请先运行统计预测")
            else:
                st.info("🔄 优化求解中...")
                progress_bar = st.progress(0)
                
                model, schedule_df = optimize_schedule(
                    st.session_state.predictions,
                    vehicle_count,
                    initial_battery,
                    solve_time
                )
                
                for i in range(60, 90):
                    progress_bar.progress(i/100)
                    time.sleep(0.02)
                
                st.session_state.optimization_result = model
                st.session_state.schedule_data = schedule_df
                
                st.success("✅ 优化求解完成！")
                st.session_state.current_stage = "优化求解完成"
                st.session_state.progress = 90
                add_log(f"优化求解完成，生成{len(schedule_df)}个班次")

    with btn5:
        if st.button("导出排班结果"):
            if st.session_state.schedule_data is None:
                st.warning("⚠️ 请先完成优化求解")
            else:
                csv_data = st.session_state.schedule_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 点击下载排班表",
                    data=csv_data,
                    file_name=f"公交排班表_{dispatch_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                st.success("✅ 排班结果已准备好下载！")
                st.session_state.current_stage = "全部完成"
                st.session_state.progress = 100
                add_log("排班结果导出完成")

    st.divider()
    st.progress(st.session_state.progress, text=f"当前进度：{st.session_state.progress}%")
    st.divider()

    status1, status2, status3, status4, status5 = st.columns(5, gap="small")
    with status1:
        st.metric("当前阶段", st.session_state.current_stage)
    with status2:
        if st.session_state.start_time:
            elapsed = int(time.time() - st.session_state.start_time)
            st.metric("已用时间", f"{elapsed}秒")
        else:
            st.metric("已用时间", "0秒")
    with status3:
        if st.session_state.progress < 100:
            remaining = int((100 - st.session_state.progress) * 0.5)
            st.metric("预计剩余", f"{remaining}秒")
        else:
            st.metric("预计剩余", "0秒")
    with status4:
        st.metric("当前Gap", f"{st.session_state.current_gap:.2f}")
    with status5:
        st.metric("当前目标值", f"{st.session_state.current_objective:.2f}")

# -------------------------- 页面2：数据管理 --------------------------
elif page == "📊 数据管理":
    st.header("📊 数据管理模块", divider="blue")
    st.divider()

    st.subheader("班次数据管理")
    if st.session_state.timetable_data is not None:
        st.dataframe(st.session_state.timetable_data, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="下载班次表",
                data=st.session_state.timetable_data.to_csv(index=False, encoding='utf-8-sig'),
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
    st.subheader("统计数据状态")
    st.info("统计数据自动从GitHub仓库的`date`文件夹读取")
    
    # 显示统计数据加载状态
    power_df, runtime_df = load_statistical_data()
    
    col1, col2 = st.columns(2)
    with col1:
        if power_df is not None:
            st.success("✅ 电量消耗数据已加载")
            st.dataframe(power_df, use_container_width=True)
        else:
            st.error("❌ 未找到date/电量消耗.csv")
    
    with col2:
        if runtime_df is not None:
            st.success("✅ 运行时间数据已加载")
            st.dataframe(runtime_df, use_container_width=True)
        else:
            st.error("❌ 未找到date/运行时间.csv")

# -------------------------- 页面3：统计预测结果 --------------------------
elif page == "📊 统计预测结果":
    st.header("📊 统计预测结果", divider="blue")
    st.divider()
    
    if 'predictions' not in st.session_state or st.session_state.predictions is None:
        st.info("请先在「今日调度」页面点击「运行统计预测」")
    else:
        # 1. 客流预测曲线
        st.subheader(f"{line}线路 {dispatch_date.strftime('%Y-%m-%d')} 客流预测曲线")
        flow_chart_data = pd.DataFrame({
            "时间": [f"{h}:00" for h in st.session_state.prediction_hours],
            "预测客流": st.session_state.predictions
        })
        st.line_chart(flow_chart_data, x="时间", y="预测客流", use_container_width=True, color="#1f77b4")
        
        st.divider()
        
        # 2. 电量消耗预测曲线
        st.subheader(f"{line}线路 {dispatch_date.strftime('%Y-%m-%d')} 电量消耗预测曲线")
        power_chart_data = pd.DataFrame({
            "时间": [f"{h}:00" for h in st.session_state.prediction_hours],
            "预测电量消耗(%)": st.session_state.power_prediction
        })
        st.line_chart(power_chart_data, x="时间", y="预测电量消耗(%)", use_container_width=True, color="#ff7f0e")
        
        st.divider()
        
        # 3. 运行时间预测曲线
        st.subheader(f"{line}线路 {dispatch_date.strftime('%Y-%m-%d')} 运行时间预测曲线")
        runtime_chart_data = pd.DataFrame({
            "时间": [f"{h}:00" for h in st.session_state.prediction_hours],
            "预测运行时间(分钟)": st.session_state.runtime_prediction
        })
        st.line_chart(runtime_chart_data, x="时间", y="预测运行时间(分钟)", use_container_width=True, color="#2ca02c")
        
        st.divider()
        
        # 4. 预测结果统计
        st.subheader("预测结果统计汇总")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            morning_peak = max(st.session_state.predictions[1:4])  # 7-9点
            st.metric("早高峰最大客流", f"{morning_peak}人")
        with col2:
            total_power = round(sum(st.session_state.power_prediction), 1)
            st.metric("总电量消耗", f"{total_power}%")
        with col3:
            avg_runtime = round(np.mean(st.session_state.runtime_prediction), 1)
            st.metric("平均运行时间", f"{avg_runtime}分钟")
        with col4:
            st.metric("当日预测天气", st.session_state.weather_data['weather'])

        # 导出所有预测结果
        st.divider()
        if st.button("📥 导出全部预测结果"):
            all_pred_data = pd.DataFrame({
                "时段": [f"{h}:00" for h in st.session_state.prediction_hours],
                "天气": [st.session_state.weather_data['weather']]*len(st.session_state.prediction_hours),
                "预测客流": st.session_state.predictions,
                "预测电量消耗(%)": st.session_state.power_prediction,
                "预测运行时间(分钟)": st.session_state.runtime_prediction
            })
            csv_data = all_pred_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="点击下载完整预测表",
                data=csv_data,
                file_name=f"统计预测结果_{dispatch_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# -------------------------- 页面4：优化求解 --------------------------
elif page == "⚙️ 优化求解":
    st.header("⚙️ 优化求解过程监控", divider="blue")
    st.divider()

    if st.session_state.optimization_result is None:
        st.info("请先在「今日调度」页面点击「开始优化求解」")
    else:
        st.subheader("求解参数")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("求解时间上限", f"{solve_time}秒")
        with col2:
            st.metric("迭代次数", f"{st.session_state.optimization_result.IterCount}次")
        with col3:
            st.metric("当前最优解", f"{st.session_state.optimization_result.ObjVal:.2f}")

        st.divider()
        st.subheader("目标函数收敛曲线")
        if len(st.session_state.convergence_data) > 0:
            iterations, objectives = zip(*st.session_state.convergence_data)
            chart_data = pd.DataFrame({
                "迭代次数": iterations,
                "目标函数值": objectives
            })
            st.line_chart(chart_data, x="迭代次数", y="目标函数值", use_container_width=True)
        else:
            st.info("收敛曲线将在求解过程中动态生成")

        st.divider()
        st.subheader("求解日志")
        log_text = "\n".join(st.session_state.solve_log)
        st.text_area("求解日志", log_text, height=300, disabled=True)

# -------------------------- 页面5：排班结果 --------------------------
elif page == "📋 排班结果":
    st.header("📋 最终排班结果", divider="blue")
    st.divider()

    if st.session_state.schedule_data is None:
        st.info("请先在「今日调度」页面完成优化求解")
    else:
        st.subheader("排班数据统计汇总")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_shifts = len(st.session_state.schedule_data)
            st.metric("总班次", f"{total_shifts}班")
        with col2:
            used_vehicles = st.session_state.schedule_data["车辆编号"].nunique()
            st.metric("使用车辆数", f"{used_vehicles}辆")
        with col3:
            avg_interval = round(16*60 / total_shifts, 1)
            st.metric("平均发车间隔", f"{avg_interval}分钟")
        with col4:
            st.metric("总运营时长", "16小时")

        st.divider()
        st.subheader("详细排班表")
        st.dataframe(st.session_state.schedule_data, use_container_width=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.schedule_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载完整排班表",
                data=csv_data,
                file_name=f"公交排班表_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("🖨️ 打印排班表"):
                st.components.v1.html("""<script>window.print();</script>""", height=0)
                st.success("✅ 打印对话框已打开")
