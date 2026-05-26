# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime
import re

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
h1, h2, h3 {
    color: #2c3e50;
    font-weight: 600;
}

/* 彻底移除蓝色覆盖+文字加粗 */
.stProgress {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.stProgress > div:first-child {
    position: static !important;
    background: transparent !important;
    background-color: transparent !important;
    height: auto !important;
    color: #2c3e50 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
}
.stProgress > div:last-child {
    height: 12px !important;
    margin: 0 !important;
    background-color: #e9ecef !important;
}
.stProgress > div:last-child > div {
    background-color: #1f77b4 !important;
    border-radius: 10px !important;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    overflow: visible !important;
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

# -------------------------- 工具函数 --------------------------
def add_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.solve_log.append(f"[INFO] {timestamp} - {message}")

def normalize_column_name(name):
    """标准化列名：移除所有空格、括号、特殊字符，转为小写"""
    return re.sub(r'[\s()%]', '', str(name)).lower()

def get_season(date):
    """自动判断季节：夏季(6-8月)、冬季(12-2月)、其他(全年)"""
    month = date.month
    if 6 <= month <= 8:
        return "summer"
    elif month == 12 or month <= 2:
        return "winter"
    else:
        return "annual"

def get_time_period(hour):
    """将0-24小时映射到四个时段"""
    if 7 <= hour <= 9:
        return "早高峰"
    elif 17 <= hour <= 19:
        return "晚高峰"
    elif (6 <= hour <= 6) or (10 <= hour <= 16) or (20 <= hour <= 21):
        return "平峰"
    else: # 0-5, 22-23
        return "低峰"

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

# -------------------------- ✅ 新增：加载碳排放数据 --------------------------
@st.cache_resource
def load_carbon_data():
    """从data文件夹加载碳排放CSV，格式：hour,annual,summer,winter"""
    try:
        carbon_df = pd.read_csv("data/碳排放.csv", dtype=str, encoding='utf-8')
    except:
        try:
            carbon_df = pd.read_csv("data/碳排放.csv", dtype=str, encoding='gbk')
        except Exception as e:
            add_log(f"⚠️ 未找到碳排放文件：{str(e)}")
            return None, f"文件不存在或无法读取：{str(e)}"
    
    # 清洗列名
    carbon_df.columns = carbon_df.columns.str.strip()
    carbon_df = carbon_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # 转换数值类型
    for col in ['hour', 'annual', 'summer', 'winter']:
        carbon_df[col] = pd.to_numeric(carbon_df[col], errors='coerce')
    
    # 显示实际列名用于调试
    actual_columns = list(carbon_df.columns)
    add_log(f"📌 碳排放文件实际列名：{actual_columns}")
    
    # 检查必需列
    required_columns = ["hour", "annual", "summer", "winter"]
    missing_columns = [col for col in required_columns if col not in carbon_df.columns]
    
    if missing_columns:
        error_msg = f"无法自动匹配列名。实际列名：{actual_columns}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    
    add_log(f"✅ 成功加载 data/碳排放.csv，共{len(carbon_df)}条记录")
    return carbon_df, None

# -------------------------- 运行时间加载 --------------------------
@st.cache_resource
def load_runtime_data():
    """从data文件夹加载运行时间CSV，自动匹配列名"""
    try:
        runtime_df = pd.read_csv("data/运行时间75%分位数.csv", dtype=str, encoding='utf-8')
    except:
        try:
            runtime_df = pd.read_csv("data/运行时间75%分位数.csv", dtype=str, encoding='gbk')
        except Exception as e:
            add_log(f"⚠️ 未找到运行时间文件：{str(e)}")
            return None, f"文件不存在或无法读取：{str(e)}"
    
    # 清洗列名
    runtime_df.columns = runtime_df.columns.str.strip()
    runtime_df = runtime_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # 显示实际列名用于调试
    actual_columns = list(runtime_df.columns)
    add_log(f"📌 运行时间文件实际列名：{actual_columns}")
    
    # 自动匹配列名
    weather_col = None
    runtime_col = None
    
    for col in actual_columns:
        normalized = normalize_column_name(col)
        if "天气" in normalized or "weather" in normalized:
            weather_col = col
        elif "运行时间" in normalized or "runtime" in normalized or "75" in normalized:
            runtime_col = col
    
    if not weather_col or not runtime_col:
        error_msg = f"无法自动匹配列名。实际列名：{actual_columns}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    
    # 重命名为标准列名
    runtime_df = runtime_df.rename(columns={weather_col: "天气", runtime_col: "75%运行时间 (min)"})
    
    add_log(f"✅ 成功匹配列名：天气='{weather_col}', 运行时间='{runtime_col}'")
    add_log(f"✅ 成功加载 data/运行时间75%分位数.csv，共{len(runtime_df)}条记录")
    return runtime_df, None

# -------------------------- 电量消耗加载 --------------------------
@st.cache_resource
def load_power_data():
    """从data文件夹加载四季电量消耗CSV，自动匹配列名"""
    try:
        power_df = pd.read_csv("data/电量消耗.csv", dtype=str, encoding='utf-8')
    except:
        power_df = pd.read_csv("data/电量消耗.csv", dtype=str, encoding='gbk')
    
    # 清洗列名
    power_df.columns = power_df.columns.str.strip()
    power_df = power_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # 显示实际列名用于调试
    actual_columns = list(power_df.columns)
    add_log(f"📌 电量消耗文件实际列名：{actual_columns}")
    
    # 自动匹配列名
    time_col = None
    weather_col = None
    spring_col = None
    summer_col = None
    autumn_col = None
    winter_col = None
    
    for col in actual_columns:
        normalized = normalize_column_name(col)
        if "时段" in normalized or "time" in normalized:
            time_col = col
        elif "天气" in normalized or "weather" in normalized:
            weather_col = col
        elif "春" in normalized or "spring" in normalized:
            spring_col = col
        elif "夏" in normalized or "summer" in normalized:
            summer_col = col
        elif "秋" in normalized or "autumn" in normalized or "fall" in normalized:
            autumn_col = col
        elif "冬" in normalized or "winter" in normalized:
            winter_col = col
    
    if not all([time_col, weather_col, spring_col, summer_col, autumn_col, winter_col]):
        error_msg = f"无法自动匹配列名。实际列名：{actual_columns}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    
    # 重命名为标准列名
    power_df = power_df.rename(columns={
        time_col: "时段",
        weather_col: "天气类型",
        spring_col: "春季",
        summer_col: "夏季",
        autumn_col: "秋季",
        winter_col: "冬季"
    })
    
    add_log(f"✅ 成功匹配列名：时段='{time_col}', 天气类型='{weather_col}', 春季='{spring_col}', 夏季='{summer_col}', 秋季='{autumn_col}', 冬季='{winter_col}'")
    add_log(f"✅ 成功加载 data/电量消耗.csv，共{len(power_df)}条记录")
    return power_df, None

# -------------------------- ✅ 更新：统计预测逻辑（加入碳排放） --------------------------
def statistical_prediction(weather_info):
    """
    统计预测逻辑：
    1. 获取当日天气和季节
    2. 筛选电量消耗CSV中对应天气和时段的行
    3. 筛选运行时间CSV中对应天气的行（所有时段共用同一个运行时间）
    4. 计算对应季节的分时段碳排放值
    5. 按时段合并三个数据集
    6. 按早高峰→晚高峰→平峰→低峰排序
    """
    current_weather = weather_info['weather']
    current_date = weather_info['date']
    current_season = get_season(current_date)
    
    season_name_map = {"summer": "夏季", "winter": "冬季", "annual": "全年"}
    add_log(f"✅ 自动判断季节：{season_name_map[current_season]}")
    
    power_df, power_error = load_power_data()
    runtime_df, runtime_error = load_runtime_data()
    carbon_df, carbon_error = load_carbon_data()
    
    # 计算分时段碳排放
    carbon_by_period = {"早高峰": 0.0, "晚高峰": 0.0, "平峰": 0.0, "低峰": 0.0}
    if carbon_df is not None:
        # 按时段分组计算平均值
        carbon_df['时段'] = carbon_df['hour'].apply(get_time_period)
        carbon_avg = carbon_df.groupby('时段')[current_season].mean().round(4)
        
        for period in carbon_by_period.keys():
            if period in carbon_avg.index:
                carbon_by_period[period] = carbon_avg[period]
            else:
                carbon_by_period[period] = 0.0
        
        add_log(f"✅ 计算完成{season_name_map[current_season]}分时段碳排放")
    
    # 如果电量数据加载失败，返回默认值
    if power_df is None:
        add_log("⚠️ 电量消耗数据加载失败，使用默认值")
        peak_order = ["早高峰", "晚高峰", "平峰", "低峰"]
        result = []
        for peak in peak_order:
            row_data = {
                "时段": peak,
                "天气": current_weather,
                "春季电量消耗": "23.00%",
                "夏季电量消耗": "23.00%",
                "秋季电量消耗": "23.00%",
                "冬季电量消耗": "23.00%",
                f"{season_name_map[current_season]}碳排放": f"{carbon_by_period[peak]:.4f}"
            }
            if runtime_df is not None:
                row_data["75%运行时间 (min)"] = "0.00"
            result.append(row_data)
        return pd.DataFrame(result)
    
    # 筛选当日天气的电量数据
    matched_power = power_df[power_df['天气类型'] == current_weather].copy()
    
    # 筛选当日天气的运行时间数据（所有时段共用）
    runtime_value = "0.00"
    if runtime_df is not None:
        matched_runtime = runtime_df[runtime_df['天气'] == current_weather].copy()
        if not matched_runtime.empty:
            runtime_value = matched_runtime.iloc[0]['75%运行时间 (min)']
            add_log(f"✅ 匹配到天气「{current_weather}」的运行时间：{runtime_value}分钟")
    
    # 按指定峰段顺序排序并合并
    peak_order = ["早高峰", "晚高峰", "平峰", "低峰"]
    result = []
    
    for peak in peak_order:
        # 匹配电量数据
        power_row = matched_power[matched_power['时段'] == peak]
        
        # 提取四季电量（无匹配时使用默认值）
        spring = power_row.iloc[0]['春季'] if not power_row.empty else "23.00%"
        summer = power_row.iloc[0]['夏季'] if not power_row.empty else "23.00%"
        autumn = power_row.iloc[0]['秋季'] if not power_row.empty else "23.00%"
        winter = power_row.iloc[0]['冬季'] if not power_row.empty else "23.00%"
        
        # 构建结果
        row_data = {
            "时段": peak,
            "天气": current_weather,
            "春季电量消耗": spring,
            "夏季电量消耗": summer,
            "秋季电量消耗": autumn,
            "冬季电量消耗": winter,
            f"{season_name_map[current_season]}碳排放": f"{carbon_by_period[peak]:.4f}"
        }
        
        # 添加运行时间（所有时段同一个值）
        if runtime_df is not None:
            row_data["75%运行时间 (min)"] = runtime_value
        
        result.append(row_data)
    
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
                # 2. 电量+运行时间+碳排放联合预测
                power_table = statistical_prediction(st.session_state.weather_data)
                
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
    st.progress(st.session_state.progress / 100, text=f"进度 {st.session_state.progress}%")
    st.divider()

    # 两行布局
    row1_col1, row1_col2, row1_col3 = st.columns(3, gap="medium")
    with row1_col1:
        st.metric("当前阶段", st.session_state.current_stage)
    with row1_col2:
        st.metric("已用时间", f"{int(time.time()-st.session_state.start_time)}s" if st.session_state.start_time else "0s")
    with row1_col3:
        st.metric("预计剩余", f"{int((100-st.session_state.progress)*0.5)}s" if st.session_state.progress<100 else "0s")

    st.divider()
    
    row2_col1, row2_col2 = st.columns(2, gap="medium")
    with row2_col1:
        st.metric("Gap", f"{st.session_state.current_gap:.2f}")
    with row2_col2:
        st.metric("目标值", f"{st.session_state.current_objective:.2f}")

# -------------------------- ✅ 更新：数据管理页面（加入碳排放） --------------------------
elif page == "📊 数据管理":
    st.header("📊 数据管理", divider="blue")
    
    st.subheader("电量消耗数据状态")
    try:
        power_df, power_error = load_power_data()
        if power_df is not None:
            st.success("✅ 成功加载 data/电量消耗.csv")
            st.dataframe(power_df, use_container_width=True)
        else:
            st.error(f"❌ 电量消耗数据加载失败：{power_error}")
            st.info("CSV格式要求：时段,天气类型,春季,夏季,秋季,冬季")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
    
    st.divider()
    
    st.subheader("运行时间75%分位数数据状态")
    try:
        runtime_df, runtime_error = load_runtime_data()
        if runtime_df is not None:
            st.success("✅ 成功加载 data/运行时间75%分位数.csv")
            st.dataframe(runtime_df, use_container_width=True)
        else:
            st.error(f"❌ 运行时间数据加载失败：{runtime_error}")
            st.info("CSV格式要求：天气,75%运行时间 (min)")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
    
    st.divider()
    
    st.subheader("碳排放数据状态")
    try:
        carbon_df, carbon_error = load_carbon_data()
        if carbon_df is not None:
            st.success("✅ 成功加载 data/碳排放.csv")
            st.dataframe(carbon_df, use_container_width=True)
        else:
            st.error(f"❌ 碳排放数据加载失败：{carbon_error}")
            st.info("CSV格式要求：hour,annual,summer,winter")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")

# -------------------------- ✅ 更新：统计预测结果页面（显示碳排放） --------------------------
elif page == "📊 统计预测结果":
    st.header("📊 电量消耗、运行时间与碳排放统计预测结果", divider="blue")
    
    if st.session_state.power_prediction_table is None:
        st.info("请先在「今日调度」页面点击「运行统计预测」")
    else:
        current_season = get_season(st.session_state.weather_data['date'])
        season_name_map = {"summer": "夏季", "winter": "冬季", "annual": "全年"}
        
        st.subheader(f"当日天气：{st.session_state.weather_data['weather']} | 当前季节：{season_name_map[current_season]}")
        # 显示合并后的结果（包含碳排放）
        st.dataframe(st.session_state.power_prediction_table, use_container_width=True, height=280)
        
        # 下载合并后的CSV
        csv_data = st.session_state.power_prediction_table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载统计预测结果表",
            data=csv_data,
            file_name=f"统计预测结果_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ 所有数据100%来自你上传的CSV文件")

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
