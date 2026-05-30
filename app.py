# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import requests
import csv
import json
from datetime import datetime, timedelta
import re
from pathlib import Path
import sys

# -------------------------- 导入公共模块（复用命令行版本的真实求解层） --------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SOLVER_SEARCH_DIRS = [
    APP_DIR,
    APP_DIR / "greedy_ga_holiday_runs",
    PROJECT_ROOT / "greedy_ga_holiday_runs",
]
for solver_dir in SOLVER_SEARCH_DIRS:
    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))

SOLVER_IMPORT_ERROR = None
try:
    from heuristic_common import (
        Config,
        HourParam,
        Solution,
        Trip,
        decode_with_random_keys,
        fmt_time,
    )
except ImportError as exc:
    SOLVER_IMPORT_ERROR = exc

# ==================== 全局页面样式 ====================
st.set_page_config(
    page_title="智能公交调度系统",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

if SOLVER_IMPORT_ERROR is not None:
    st.error("缺少算法核心文件 heuristic_common.py，页面已启动但无法继续求解。")
    st.info("部署时请把 页面/heuristic_common.py 和 app.py 放在同一个目录；或者保留 greedy_ga_holiday_runs/heuristic_common.py 目录结构。")
    st.code("\n".join(str(path) for path in SOLVER_SEARCH_DIRS), language="text")
    st.stop()

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

# ==================== 会话状态初始化 ====================
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
if 'weather_source' not in st.session_state:
    st.session_state.weather_source = ""
# 贪心/遗传结果
if 'greedy_solution' not in st.session_state:
    st.session_state.greedy_solution = None
if 'greedy_schedule_data' not in st.session_state:
    st.session_state.greedy_schedule_data = None
if 'greedy_charge_data' not in st.session_state:
    st.session_state.greedy_charge_data = None
if 'greedy_objective' not in st.session_state:
    st.session_state.greedy_objective = 0.0
if 'charge_data' not in st.session_state:
    st.session_state.charge_data = None
# 遗传算法历史数据
if 'ga_history' not in st.session_state:
    st.session_state.ga_history = None
if 'best_chromosome' not in st.session_state:
    st.session_state.best_chromosome = None
# 记录当前选择的求解方式（跨页面使用）
if 'current_solve_mode' not in st.session_state:
    st.session_state.current_solve_mode = ""

# ==================== 工具函数 ====================
def add_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.solve_log.append(f"[INFO] {timestamp} - {message}")

def normalize_column_name(name):
    return re.sub(r'[\s()%]', '', str(name)).lower()

DATA_SEARCH_DIRS = [
    APP_DIR / "data",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "验证不等式",
]

TIMETABLE_CANDIDATES = {
    "工作日": ["工作日发车时刻表.csv", "工作日发车时刻表(1).csv", "节假日发车时刻表(1).csv"],
    "周末": ["周末发车时刻表.csv", "节假日发车时刻表.csv", "节假日发车时刻表(1).csv"],
    "节假日": ["节假日发车时刻表.csv", "节假日发车时刻表(1).csv"],
}

def resolve_data_file(candidates):
    if isinstance(candidates, str):
        candidates = [candidates]
    checked = []
    for base_dir in DATA_SEARCH_DIRS:
        for name in candidates:
            path = base_dir / name
            checked.append(str(path))
            if path.exists():
                return path
    raise FileNotFoundError("未找到数据文件，已检查：" + "；".join(checked))

def read_csv_with_fallback(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="gbk")

def parse_percent_to_fraction(value, default=0.1):
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        number = float(text)
        return number / 100.0 if number > 1 else number
    except Exception:
        return default

def parse_float_value(value, default=0.0):
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default
        return float(text.replace("%", ""))
    except Exception:
        return default

def get_carbon_season(date):
    month = date.month
    if 6 <= month <= 8:
        return "summer"
    elif month == 12 or month <= 2:
        return "winter"
    else:
        return "annual"

def get_power_season(date):
    month = date.month
    if 3 <= month <= 5:
        return "春季"
    elif 6 <= month <= 8:
        return "夏季"
    elif 9 <= month <= 11:
        return "秋季"
    else:
        return "冬季"

def get_time_period(hour):
    if 7 <= hour <= 9:
        return "早高峰"
    elif 17 <= hour <= 19:
        return "晚高峰"
    elif hour == 6 or (10 <= hour <= 16) or (20 <= hour <= 21):
        return "平峰"
    else:
        return "低峰"

# ==================== 天气获取（修复高德API无效提示） ====================
def get_weather_forecast(date):
    WEATHER_API_KEY = "e088a35c897818780a479973d4623063"
    today = datetime.now().date()
    max_forecast_date = today + timedelta(days=3)
    is_in_forecast_range = (date >= today) and (date <= max_forecast_date)
    if is_in_forecast_range:
        try:
            city_code = "110000"
            url = (
                f"https://restapi.amap.com/v3/weather/weatherInfo"
                f"?city={city_code}&key={WEATHER_API_KEY}&extensions=all"
            )
            response = requests.get(url, timeout=10)
            data = response.json()
            if data.get("status") == "1":
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
                        st.session_state.weather_source = f"✅ 高德API预报（{target}）"
                        add_log(f"✅ 高德API成功：{target} {weather_info['weather']} {weather_info['temp_min']}~{weather_info['temp_max']}℃")
                        return weather_info, None
                add_log(f"⚠️ 高德返回无 {target}（超出3天？）")
            else:
                add_log(f"⚠️ 高德API调用失败：{data.get('info')}，将使用默认天气")
        except Exception as e:
            add_log(f"⚠️ 高德请求异常：{e}，将使用默认天气")
    else:
        st.session_state.weather_source = "ℹ️ 历史/超3天 → 使用默认天气"
        add_log(f"ℹ️ {date} 不在未来3天 → 使用默认天气")
    default_weather = {
        "date": date,
        "temp_max": 25,
        "temp_min": 18,
        "weather": "晴",
        "is_rain": 0
    }
    return default_weather, None

# ==================== 加载时刻表（完全兼容heuristic_common.py格式） ====================
@st.cache_resource
def load_timetable_data(timetable_type):
    try:
        file_path = resolve_data_file(TIMETABLE_CANDIDATES.get(timetable_type, TIMETABLE_CANDIDATES["工作日"]))
        add_log(f"🔄 正在读取 {timetable_type} 时刻表：{file_path}")
        df = read_csv_with_fallback(file_path)
    except Exception as e:
        add_log(f"⚠️ 未找到 {timetable_type} 时刻表文件：{str(e)}")
        return None, str(e)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.dropna(how='all')
    add_log(f"✅ 成功加载原始时刻表，共{len(df)}行")
    add_log(f"📌 原始列名：{list(df.columns)}")
    all_trips = []
    for col in df.columns:
        normalized = normalize_column_name(col)
        if "发车时刻" in normalized or "发车时间" in normalized:
            if "四惠" in col and "老山" in col:
                sihui_pos = col.index("四惠")
                laoshan_pos = col.index("老山")
                direction = "四惠" if sihui_pos < laoshan_pos else "老山"
            elif "四惠" in col:
                direction = "四惠"
            elif "老山" in col:
                direction = "老山"
            else:
                direction = "四惠"
            add_log(f"✅ 识别到列：{col} → 方向：{direction}")
            times = df[col].dropna().tolist()
            for time_str in times:
                parts = time_str.split(":")
                if len(parts) == 2:
                    try:
                        depart_hour = int(parts[0])
                        depart_minute = int(parts[1])
                        all_trips.append({
                            "depart_time": time_str,
                            "depart_hour": depart_hour,
                            "depart_minute": depart_minute,
                            "direction": direction
                        })
                    except:
                        pass
    if not all_trips:
        error_msg = "未找到任何发车时刻列，请检查CSV文件列名"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    direction_count = {}
    for trip in all_trips:
        direction_count[trip["direction"]] = direction_count.get(trip["direction"], 0) + 1
    add_log(f"✅ 方向统计：{direction_count}")
    add_log(f"✅ 合并完成，共{len(all_trips)}个有效发车班次")
    return all_trips, None

# ==================== 加载各类原始CSV ====================
@st.cache_resource
def load_carbon_data():
    try:
        file_path = resolve_data_file("碳排放.csv")
        carbon_df = read_csv_with_fallback(file_path)
    except Exception as e:
        return None, str(e)
    carbon_df.columns = carbon_df.columns.str.strip()
    carbon_df = carbon_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    for col in ['hour', 'annual', 'summer', 'winter']:
        carbon_df[col] = pd.to_numeric(carbon_df[col], errors='coerce')
    required_columns = ["hour", "annual", "summer", "winter"]
    missing_columns = [col for col in required_columns if col not in carbon_df.columns]
    if missing_columns:
        error_msg = f"无法自动匹配列名。实际列名：{list(carbon_df.columns)}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    add_log(f"✅ 成功加载 {file_path}，共{len(carbon_df)}条记录")
    return carbon_df, None

@st.cache_resource
def load_runtime_data():
    try:
        file_path = resolve_data_file("运行时间75%分位数.csv")
        runtime_df = read_csv_with_fallback(file_path)
    except Exception as e:
        return None, str(e)
    runtime_df.columns = runtime_df.columns.str.strip()
    runtime_df = runtime_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    weather_col = None
    runtime_col = None
    for col in runtime_df.columns:
        normalized = normalize_column_name(col)
        if "天气" in normalized or "weather" in normalized:
            weather_col = col
        elif "运行时间" in normalized or "runtime" in normalized or "75" in normalized:
            runtime_col = col
    if not weather_col or not runtime_col:
        error_msg = f"无法自动匹配列名。实际列名：{list(runtime_df.columns)}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    runtime_df = runtime_df.rename(columns={weather_col: "天气", runtime_col: "75%运行时间 (min)"})
    add_log(f"✅ 成功匹配列名：天气='{weather_col}', 运行时间='{runtime_col}'")
    add_log(f"✅ 成功加载 {file_path}，共{len(runtime_df)}条记录")
    return runtime_df, None

@st.cache_resource
def load_power_data():
    try:
        file_path = resolve_data_file("电量消耗.csv")
        power_df = read_csv_with_fallback(file_path)
    except Exception as e:
        return None, str(e)
    power_df.columns = power_df.columns.str.strip()
    power_df = power_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    time_col = None
    weather_col = None
    spring_col = None
    summer_col = None
    autumn_col = None
    winter_col = None
    for col in power_df.columns:
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
        error_msg = f"无法自动匹配列名。实际列名：{list(power_df.columns)}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    power_df = power_df.rename(columns={
        time_col: "时段",
        weather_col: "天气类型",
        spring_col: "春季",
        summer_col: "夏季",
        autumn_col: "秋季",
        winter_col: "冬季"
    })
    add_log(f"✅ 成功匹配列名：时段='{time_col}', 天气类型='{weather_col}', 春季='{spring_col}', 夏季='{summer_col}', 秋季='{autumn_col}', 冬季='{winter_col}'")
    add_log(f"✅ 成功加载 {file_path}，共{len(power_df)}条记录")
    return power_df, None

@st.cache_resource
def load_hourly_template_data():
    try:
        file_path = resolve_data_file("2026-05-26T06-59_export (1).csv")
        template_df = read_csv_with_fallback(file_path)
    except Exception as e:
        return None, str(e)
    template_df.columns = template_df.columns.str.strip()
    add_log(f"✅ 成功加载逐时参数模板 {file_path}，共{len(template_df)}条记录")
    return template_df, None

# ==================== 统计预测、客流预测 ====================
def statistical_prediction(weather_info):
    current_weather = weather_info['weather']
    current_date = weather_info['date']
    power_season = get_power_season(current_date)
    carbon_season = get_carbon_season(current_date)
    season_name_map = {"summer": "夏季", "winter": "冬季", "annual": "全年"}
    add_log(f"✅ 自动判断：电量季节={power_season}, 碳排放季节={season_name_map[carbon_season]}")
    power_df, power_error = load_power_data()
    runtime_df, runtime_error = load_runtime_data()
    carbon_df, carbon_error = load_carbon_data()
    template_df, template_error = load_hourly_template_data()
    template_rows = {}
    template_power_col = None
    template_runtime_col = None
    template_carbon_col = None
    if template_df is not None:
        template_power_col = next((c for c in template_df.columns if "电量消耗" in c), None)
        template_runtime_col = next((c for c in template_df.columns if "运行时间" in c), None)
        template_carbon_col = next((c for c in template_df.columns if "碳排放" in c), None)
        for _, row in template_df.iterrows():
            hour_value = row["小时"] if "小时" in template_df.columns else row.iloc[0]
            try:
                hour = int(str(hour_value).split(":")[0])
                template_rows[hour] = row
            except Exception:
                continue
    runtime_value = "0.00"
    if runtime_df is not None:
        matched_runtime = runtime_df[runtime_df['天气'] == current_weather].copy()
        if not matched_runtime.empty:
            runtime_value = matched_runtime.iloc[0]['75%运行时间 (min)']
            add_log(f"✅ 匹配到天气「{current_weather}」的运行时间：{runtime_value}分钟")
    matched_power = None
    if power_df is not None:
        matched_power = power_df[power_df['天气类型'] == current_weather].copy()
        add_log(f"✅ 匹配到天气「{current_weather}」的电量消耗数据")
    result = []
    power_column_name = f"{power_season}电量消耗"
    carbon_column_name = f"{season_name_map[carbon_season]}碳排放"
    for hour in range(0, 24):
        period = get_time_period(hour)
        template_row = template_rows.get(hour)
        runtime_value_hour = runtime_value
        if runtime_value_hour == "0.00" and template_row is not None and template_runtime_col:
            runtime_value_hour = template_row[template_runtime_col]
        power_value = "23.00%"
        if template_row is not None and template_power_col:
            power_value = template_row[template_power_col]
        if matched_power is not None:
            power_row = matched_power[matched_power['时段'] == period]
            if not power_row.empty:
                power_value = power_row.iloc[0][power_season]
        carbon_value = 0.0
        if template_row is not None and template_carbon_col:
            carbon_value = parse_float_value(template_row[template_carbon_col], default=0.0)
        if carbon_df is not None:
            carbon_row = carbon_df[carbon_df['hour'] == hour]
            if not carbon_row.empty:
                carbon_value = carbon_row.iloc[0][carbon_season]
        row_data = {
            "小时": f"{hour:02d}:00",
            "时段类型": period,
            "天气": current_weather,
            power_column_name: power_value,
            "75%运行时间 (min)": runtime_value_hour,
            "碳排放量": f"{carbon_value:.4f}"
        }
        result.append(row_data)
    return pd.DataFrame(result)

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
        predictions.append(round(flow * (0.9 + random.random() * 0.2)))
    return hours, predictions

# ==================== 从统计预测表解析参数（直接传给heuristic_common.py） ====================
def build_hour_params_from_pred_table(pred_df):
    hour_params = {}
    run_col = "75%运行时间 (min)"
    pwr_col = next((c for c in pred_df.columns if "电量消耗" in c), None)
    carbon_col = next((c for c in pred_df.columns if "碳排放" in c), None)
    if pwr_col is None:
        raise ValueError("统计预测表缺少“电量消耗”列，无法构造求解参数")
    if carbon_col is None:
        raise ValueError("统计预测表缺少“碳排放”列，无法构造求解参数")
    for _, row in pred_df.iterrows():
        hour_str = row["小时"]
        hour = int(hour_str.split(":")[0])
        hour_params[hour] = HourParam(
            hour=hour,
            energy_fraction=parse_percent_to_fraction(row[pwr_col], default=0.1),
            runtime_min=parse_float_value(row[run_col], default=45.0),
            carbon_factor=parse_float_value(row[carbon_col], default=0.0),
        )
    missing = sorted(set(range(24)) - set(hour_params))
    if missing:
        raise ValueError(f"统计预测表缺少小时参数：{missing}")
    add_log("✅ 优化求解：已转换为 heuristic_common.HourParam 参数")
    return hour_params

def build_trips_for_solver(raw_trips, hour_params):
    if not raw_trips:
        raise ValueError("班次表为空，无法求解")
    normalized = []
    for item in raw_trips:
        depart_time = str(item.get("depart_time", "")).strip()
        if ":" in depart_time:
            hour_str, minute_str = depart_time.split(":", 1)
            depart_hour = int(hour_str)
            depart_minute = int(minute_str)
        else:
            depart_hour = int(item["depart_hour"])
            depart_minute = int(item["depart_minute"])
        direction = str(item.get("direction", "四惠"))
        if direction == "B" or direction.startswith("老山"):
            origin, dest = "B", "A"
        else:
            origin, dest = "A", "B"
        depart_min = depart_hour * 60 + depart_minute
        normalized.append((origin, dest, depart_min))
    normalized.sort(key=lambda row: (row[2], row[0], row[1]))
    trips = []
    for idx, (origin, dest, depart_min) in enumerate(normalized):
        hour = int(depart_min // 60) % 24
        hp = hour_params[hour]
        trips.append(
            Trip(
                id=idx,
                origin=origin,
                dest=dest,
                depart_min=depart_min,
                hour=hour,
                runtime_min=hp.runtime_min,
                energy_fraction=hp.energy_fraction,
                carbon_factor=hp.carbon_factor,
                label=f"T{idx + 1:03d}_{origin}_to_{dest}_{int(depart_min // 60):02d}{int(depart_min % 60):02d}",
            )
        )
    add_log(f"✅ 优化求解：已转换为 heuristic_common.Trip 班次，共{len(trips)}个")
    return trips

def make_solver_config(vehicle_count):
    total = int(vehicle_count)
    inventory_a = total // 2
    inventory_b = total - inventory_a
    return Config(
        real_total_vehicles=total,
        initial_inventory={"A": inventory_a, "B": inventory_b},
        min_end_inventory={"A": min(40, inventory_a), "B": min(40, inventory_b)},
        charger_capacity={"A": 40, "B": 40},
        rest_minutes=25.0,
        max_late_minutes=5.0,
    )

def solution_to_trip_dataframe(solution):
    fields = [
        "vehicle_id", "sequence", "trip_id", "trip_label", "origin", "dest",
        "scheduled_depart", "actual_depart", "arrival", "late_min",
        "runtime_min", "energy_fraction", "soc_before", "soc_after",
    ]
    rows = []
    for vehicle in solution.routes:
        trip_seq = 0
        activities = sorted(vehicle.activities, key=lambda item: (item.get("start_min", item.get("actual_depart", 0)), item["type"]))
        for act in activities:
            if act["type"] != "trip":
                continue
            trip_seq += 1
            rows.append({
                "vehicle_id": vehicle.id,
                "sequence": trip_seq,
                "trip_id": act["trip_id"] + 1,
                "trip_label": act["trip_label"],
                "origin": act["origin"],
                "dest": act["dest"],
                "scheduled_depart": fmt_time(act["scheduled_depart"]),
                "actual_depart": fmt_time(act["actual_depart"]),
                "arrival": fmt_time(act["arrival_min"]),
                "late_min": round(act["late_min"], 6),
                "runtime_min": round(act["runtime_min"], 6),
                "energy_fraction": round(act["energy_fraction"], 8),
                "soc_before": round(act["soc_before"], 8),
                "soc_after": round(act["soc_after"], 8),
            })
    return pd.DataFrame(rows, columns=fields)

def solution_to_charge_dataframe(solution):
    fields = ["vehicle_id", "sequence", "kind", "endpoint", "start", "end", "q", "cost", "soc_before", "soc_after", "slot_ids"]
    rows = []
    for vehicle in solution.routes:
        charge_seq = 0
        activities = sorted(vehicle.activities, key=lambda item: (item.get("start_min", item.get("actual_depart", 0)), item["type"]))
        for act in activities:
            if act["type"] not in {"op_charge", "post_charge"}:
                continue
            charge_seq += 1
            rows.append({
                "vehicle_id": vehicle.id,
                "sequence": charge_seq,
                "kind": act["type"],
                "endpoint": act["endpoint"],
                "start": fmt_time(act["start_min"]),
                "end": fmt_time(act["end_min"]),
                "q": round(act["q"], 8),
                "cost": round(act["cost"], 8),
                "soc_before": "" if act.get("soc_before") is None else round(act["soc_before"], 8),
                "soc_after": "" if act.get("soc_after") is None else round(act["soc_after"], 8),
                "slot_ids": " ".join(str(gid) for gid in act.get("slot_ids", [])),
            })
    return pd.DataFrame(rows, columns=fields)

# ==================== 遗传算子（与命令行版本1:1完全一致） ====================
def tournament(rng: random.Random, scored: list[tuple[float, list[float], Solution]], size: int = 3) -> list[float]:
    picks = [rng.choice(scored) for _ in range(size)]
    picks.sort(key=lambda item: item[0])
    return picks[0][1]

def crossover(rng: random.Random, left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("Chromosome length mismatch.")
    if len(left) <= 2:
        return left[:]
    a = rng.randrange(1, len(left) - 1)
    b = rng.randrange(a, len(left))
    child = left[:a] + right[a:b] + left[b:]
    if rng.random() < 0.25:
        child = [right[i] if rng.random() < 0.5 else child[i] for i in range(len(child))]
    return child

def mutate(rng: random.Random, chromosome: list[float], rate: float) -> None:
    for i in range(len(chromosome)):
        if rng.random() < rate:
            chromosome[i] = rng.random()

def fitness(solution: Solution) -> float:
    return solution.objective

# ==================== 优化主函数（1:1兼容命令行参数 + 仅Gap收敛） ====================
def optimize_greedy_only(trips, hour_params, config, initial_battery, power_prediction_table):
    """仅执行贪心算法（与命令行greedy.py完全一致）"""
    add_log("🔄 运行粗略求解（贪心算法）")
    greedy_solution = decode_with_random_keys(trips, hour_params, config, algorithm="greedy")
    trip_df = solution_to_trip_dataframe(greedy_solution)
    charge_df = solution_to_charge_dataframe(greedy_solution)
    return greedy_solution, trip_df, charge_df

def optimize_genetic_full(trips, hour_params, config, initial_battery, power_prediction_table):
    """执行遗传算法（与命令行run_genetic.py参数1:1一致）"""
    add_log("🔄 运行精确求解（遗传算法）")
    run_started = time.perf_counter()
    seed = 20260528
    rng = random.Random(seed)
    n = len(trips)
    pop_size = 56
    generations = 90
    elite_num = 6
    mut_rate = 0.055
    top_k = 5

    population: list[list[float]] = [[0.0 for _ in range(n)]]
    while len(population) < pop_size:
        population.append([rng.random() for _ in range(n)])

    best_solution: Solution | None = None
    best_chromosome: list[float] | None = None
    ga_history = []
    st.session_state.convergence_data = []

    for current_gen in range(generations + 1):
        scored: list[tuple[float, list[float], Solution]] = []
        for chrom in population:
            sol = decode_with_random_keys(
                trips,
                hour_params,
                config,
                genes=chrom,
                top_k=top_k,
                algorithm="genetic",
            )
            scored.append((fitness(sol), chrom, sol))
        scored.sort(key=lambda item: item[0])

        if best_solution is None or scored[0][0] < fitness(best_solution):
            best_solution = scored[0][2]
            best_chromosome = scored[0][1][:]

        best = scored[0][2]
        feasible_count = sum(1 for _, _, sol in scored if sol.feasible)
        st.session_state.convergence_data.append((current_gen, best.objective))
        ga_history.append({
            "generation": current_gen,
            "best_objective": best.objective,
            "best_feasible": best.feasible,
            "best_vehicles": best.vehicles_used,
            "best_gap_to_lb": best.relative_gap_to_lb,
            "best_late_min": best.total_late_min,
            "feasible_count": feasible_count,
            "population": len(population),
        })
        st.session_state.current_gap = best.relative_gap_to_lb

        add_log(
            f"gen={current_gen:03d} best={best.objective:.6f} feasible={best.feasible} "
            f"vehicles={best.vehicles_used} gap_lb={best.relative_gap_to_lb:.6f} feasible_count={feasible_count}"
        )

        if current_gen == generations:
            break

        next_population: list[list[float]] = [chrom[:] for _, chrom, _ in scored[: elite_num]]
        while len(next_population) < pop_size:
            left = tournament(rng, scored)
            right = tournament(rng, scored)
            child = crossover(rng, left, right)
            mutate(rng, child, mut_rate)
            next_population.append(child)
        population = next_population

    if best_solution is None or best_chromosome is None:
        raise RuntimeError("遗传算法没有产生可用解")
    total_runtime_sec = time.perf_counter() - run_started
    best_solution.runtime_sec = total_runtime_sec
    best_solution.metadata["ga_parameters"] = {
        "population": pop_size,
        "generations": generations,
        "elite": elite_num,
        "mutation_rate": mut_rate,
        "top_k": top_k,
        "seed": seed,
    }
    st.session_state.ga_history = ga_history
    st.session_state.best_chromosome = best_chromosome
    st.session_state.current_objective = best_solution.objective
    trip_df = solution_to_trip_dataframe(best_solution)
    charge_df = solution_to_charge_dataframe(best_solution)
    return best_solution, trip_df, charge_df

# ==================== 侧边栏 & 页面布局 ====================
st.sidebar.title("🚌 智能公交调度系统")
st.sidebar.markdown("""<style>[data-testid="stSidebar"] {background-color: #f0f5fa;}</style>""", unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("功能模块", ["📅 今日调度", "📊 数据管理", "📊 统计预测结果", "⚙️ 优化求解", "📋 排班结果"])
st.sidebar.divider()
st.sidebar.info("智能公交调度系统")

# -------------------------- 今日调度页面 --------------------------
if page == "📅 今日调度":
    st.header("🚌 智能公交调度", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        dispatch_date = st.date_input("调度日期", datetime.now().date())
        line = st.selectbox("线路/场站", ["1路", "2路", "3路", "4路", "5路"])
        timetable_type = st.selectbox("班次表", ["工作日", "周末", "节假日"])
    with col2:
        vehicle_count = st.number_input("当日车辆数", 1, 120, 87)
        initial_battery = st.number_input("初始电量（%）", 0, 100, 100)
        solve_time = st.number_input("求解时间上限（秒）", 60, 3600, 300)

    # 求解方式选择
    st.divider()
    solve_mode = st.selectbox("优化求解方式", ["粗略求解（贪心算法）", "精确求解（遗传算法）"])
    # 保存当前选择的模式，供其他页面判断
    st.session_state.current_solve_mode = solve_mode
    st.divider()

    btn1, btn2, btn3, btn4, btn5 = st.columns(5, gap="small")
    with btn1:
        if st.button("读取班次表"):
            st.session_state.start_time = time.time()
            timetable_df, timetable_error = load_timetable_data(timetable_type)
            if timetable_df is not None:
                st.session_state.timetable_data = timetable_df
                st.success(f"✅ 成功读取 {timetable_type} 班次表，共{len(timetable_df)}条记录")
            else:
                st.session_state.timetable_data = [
                    {"depart_time": f"{6+i//2:02d}:{i%2*30:02d}",
                     "depart_hour": 6+i//2,
                     "depart_minute": i%2*30,
                     "direction": "四惠" if i%2==0 else "老山"}
                    for i in range(10)
                ]
                st.warning(f"⚠️ 未找到 {timetable_type} 班次表，使用示例数据")
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
                hours, preds = predict_passenger_flow(dispatch_date, line, is_workday, st.session_state.weather_data)
                power_table = statistical_prediction(st.session_state.weather_data)
                st.session_state.predictions = preds
                st.session_state.prediction_hours = hours
                st.session_state.power_prediction_table = power_table
                st.session_state.progress = 60
                st.session_state.current_stage = "统计预测完成"
                st.success("✅ 统计预测完成！")

    with btn4:
        if st.button("开始优化求解"):
            # 安全判断
            predictions_ok = st.session_state.predictions is not None and len(st.session_state.predictions) > 0
            table_ok = st.session_state.power_prediction_table is not None and not st.session_state.power_prediction_table.empty
            if not predictions_ok or not table_ok:
                st.warning("⚠️ 请先运行统计预测")
            else:
                st.info("🔄 求解中...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                hour_params = build_hour_params_from_pred_table(st.session_state.power_prediction_table)
                trips = build_trips_for_solver(st.session_state.timetable_data, hour_params)
                config = make_solver_config(vehicle_count)

                # 根据选择执行对应算法
                if solve_mode == "粗略求解（贪心算法）":
                    status_text.text("正在运行粗略求解（贪心算法）...")
                    progress_bar.progress(10)
                    greedy_sol, greedy_df, greedy_charge_df = optimize_greedy_only(trips, hour_params, config, initial_battery, st.session_state.power_prediction_table)
                    st.session_state.greedy_solution = greedy_sol
                    st.session_state.greedy_schedule_data = greedy_df
                    st.session_state.greedy_charge_data = greedy_charge_df
                    st.session_state.greedy_objective = greedy_sol.objective
                    st.session_state.optimization_result = None
                    st.session_state.schedule_data = None
                    st.session_state.charge_data = None
                    st.session_state.current_objective = greedy_sol.objective
                    progress_bar.progress(100)
                    status_text.empty()
                    st.success("✅ 粗略求解完成！")
                else:
                    # 精确求解：先跑贪心，再跑遗传（与命令行逻辑一致）
                    status_text.text("正在运行贪心算法（基准解）...")
                    progress_bar.progress(5)
                    greedy_sol, greedy_df, greedy_charge_df = optimize_greedy_only(trips, hour_params, config, initial_battery, st.session_state.power_prediction_table)
                    st.session_state.greedy_solution = greedy_sol
                    st.session_state.greedy_schedule_data = greedy_df
                    st.session_state.greedy_charge_data = greedy_charge_df
                    st.session_state.greedy_objective = greedy_sol.objective

                    status_text.text("正在运行遗传算法（精确解）...")
                    progress_bar.progress(10)
                    gen_sol, gen_df, gen_charge_df = optimize_genetic_full(trips, hour_params, config, initial_battery, st.session_state.power_prediction_table)
                    st.session_state.optimization_result = gen_sol
                    st.session_state.schedule_data = gen_df
                    st.session_state.charge_data = gen_charge_df
                    progress_bar.progress(100)
                    status_text.empty()
                    st.success("✅ 精确求解完成！")

                st.session_state.progress = 90
                st.session_state.current_stage = "优化求解完成"

    with btn5:
        if st.button("导出排班结果"):
            if st.session_state.greedy_schedule_data is not None:
                dispatch_date = st.session_state.weather_data["date"] if st.session_state.weather_data else datetime.now().date()
                csv_greedy = st.session_state.greedy_schedule_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 下载粗略解排班表", csv_greedy, f"公交排班表_粗略解_{dispatch_date.strftime('%Y%m%d')}.csv")
                if st.session_state.greedy_charge_data is not None:
                    csv_greedy_charge = st.session_state.greedy_charge_data.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载粗略解充电表", csv_greedy_charge, f"公交充电表_粗略解_{dispatch_date.strftime('%Y%m%d')}.csv")
                # 仅遗传模式才导出遗传相关文件
                if st.session_state.current_solve_mode == "精确求解（遗传算法）" and st.session_state.schedule_data is not None:
                    csv_genetic = st.session_state.schedule_data.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载精确解排班表", csv_genetic, f"公交排班表_精确解_{dispatch_date.strftime('%Y%m%d')}.csv")
                    if st.session_state.charge_data is not None:
                        csv_genetic_charge = st.session_state.charge_data.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button("📥 下载精确解充电表", csv_genetic_charge, f"公交充电表_精确解_{dispatch_date.strftime('%Y%m%d')}.csv")
                    if st.session_state.ga_history:
                        hist_df = pd.DataFrame(st.session_state.ga_history)
                        csv_hist = hist_df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("📥 下载遗传迭代历史", csv_hist, f"GA_历史记录_{dispatch_date.strftime('%Y%m%d')}.csv")
                st.session_state.progress = 100
                st.session_state.current_stage = "全部完成"
            else:
                st.warning("⚠️ 请先完成求解")

    st.divider()
    st.progress(st.session_state.progress / 100, text=f"进度 {st.session_state.progress}%")
    st.divider()
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
        st.metric("当前收敛Gap", f"{st.session_state.current_gap:.4f}")
    with row2_col2:
        st.metric("目标值", f"{st.session_state.current_objective:.2f}")

# -------------------------- 数据管理页面 --------------------------
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
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
    st.divider()
    st.subheader("班次表数据状态")
    try:
        if st.session_state.timetable_data is not None:
            st.success("✅ 已加载班次表数据（保留方向信息）")
            df = pd.DataFrame(st.session_state.timetable_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("请在「今日调度」页面点击「读取班次表」加载数据")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")

# -------------------------- 统计预测结果页面 --------------------------
elif page == "📊 统计预测结果":
    st.header("📊 24小时逐时统计预测结果", divider="blue")
    if st.session_state.power_prediction_table is None or st.session_state.power_prediction_table.empty:
        st.info("请先在「今日调度」页面点击「运行统计预测」")
    else:
        current_date = st.session_state.weather_data['date'] if st.session_state.weather_data else datetime.now().date()
        power_season = get_power_season(current_date)
        carbon_season = get_carbon_season(current_date)
        season_name_map = {"summer": "夏季", "winter": "冬季", "annual": "全年"}
        st.subheader(f"调度日期：{current_date.strftime('%Y-%m-%d')} | 当日天气：{st.session_state.weather_data['weather'] if st.session_state.weather_data else '无'}")
        st.subheader(f"电量季节：{power_season} | 碳排放季节：{season_name_map[carbon_season]}")
        st.dataframe(st.session_state.power_prediction_table, use_container_width=True, height=800)
        csv_data = st.session_state.power_prediction_table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 下载24小时逐时统计预测结果表", csv_data, f"24小时逐时统计预测结果_{current_date.strftime('%Y%m%d')}.csv")
        st.success("✅ 所有数据来自上传CSV文件，匹配当日天气和季节")

# -------------------------- 优化求解页面（区分两种模式） --------------------------
elif page == "⚙️ 优化求解":
    st.header("⚙️ 优化求解", divider="blue")
    solve_mode = st.session_state.current_solve_mode

    # 粗略求解：只展示贪心结果
    if solve_mode == "粗略求解（贪心算法）":
        if st.session_state.greedy_solution:
            st.subheader("📌 粗略解（贪心算法）")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最优目标值", f"{st.session_state.greedy_objective:.2f}")
            with col2:
                st.metric("使用车辆数", st.session_state.greedy_solution.vehicles_used)
        else:
            st.info("请先在「今日调度」页面点击「开始优化求解」")

    # 精确求解：同时展示贪心 + 遗传
    elif solve_mode == "精确求解（遗传算法）":
        if st.session_state.greedy_solution:
            st.subheader("📌 基准解（贪心算法）")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最优目标值", f"{st.session_state.greedy_objective:.2f}")
            with col2:
                st.metric("使用车辆数", st.session_state.greedy_solution.vehicles_used)
            st.divider()

        if st.session_state.optimization_result:
            st.subheader("🎯 精确解（遗传算法）")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最优目标值", f"{st.session_state.current_objective:.2f}")
            with col2:
                st.metric("使用车辆数", st.session_state.optimization_result.vehicles_used)
            with col3:
                st.metric("运行耗时(s)", f"{st.session_state.optimization_result.runtime_sec:.2f}")
            if st.session_state.convergence_data:
                st.subheader("遗传算法收敛曲线")
                conv_df = pd.DataFrame(st.session_state.convergence_data, columns=["代次", "目标值"])
                st.line_chart(conv_df.set_index("代次"))
            if st.session_state.ga_history:
                st.subheader("每代迭代明细")
                hist_df = pd.DataFrame(st.session_state.ga_history)
                st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("请先在「今日调度」页面点击「开始优化求解」")

# -------------------------- 排班结果页面（区分两种模式） --------------------------
elif page == "📋 排班结果":
    st.header("📋 排班结果", divider="blue")
    solve_mode = st.session_state.current_solve_mode

    # 粗略求解：仅展示贪心排班表
    if solve_mode == "粗略求解（贪心算法）":
        if st.session_state.greedy_schedule_data is not None:
            st.subheader("📌 粗略解（贪心算法）排班表")
            st.dataframe(st.session_state.greedy_schedule_data, use_container_width=True)
            csv_greedy = st.session_state.greedy_schedule_data.to_csv(index=False, encoding='utf-8-sig')
            current_date = st.session_state.weather_data["date"] if st.session_state.weather_data else datetime.now().date()
            st.download_button("📥 下载粗略解排班表", csv_greedy, f"公交排班表_粗略解_{current_date.strftime('%Y%m%d')}.csv")
            if st.session_state.greedy_charge_data is not None:
                with st.expander("粗略解充电表"):
                    st.dataframe(st.session_state.greedy_charge_data, use_container_width=True)
                    csv_greedy_charge = st.session_state.greedy_charge_data.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载粗略解充电表", csv_greedy_charge, f"公交充电表_粗略解_{current_date.strftime('%Y%m%d')}.csv")
        else:
            st.info("请先完成优化求解")

    # 精确求解：同时展示两个排班表
    elif solve_mode == "精确求解（遗传算法）":
        if st.session_state.greedy_schedule_data is not None:
            st.subheader("📌 基准解（贪心算法）排班表")
            st.dataframe(st.session_state.greedy_schedule_data, use_container_width=True)
            csv_greedy = st.session_state.greedy_schedule_data.to_csv(index=False, encoding='utf-8-sig')
            current_date = st.session_state.weather_data["date"] if st.session_state.weather_data else datetime.now().date()
            st.download_button("📥 下载基准解排班表", csv_greedy, f"公交排班表_基准解_{current_date.strftime('%Y%m%d')}.csv")
            if st.session_state.greedy_charge_data is not None:
                with st.expander("基准解充电表"):
                    st.dataframe(st.session_state.greedy_charge_data, use_container_width=True)
                    csv_greedy_charge = st.session_state.greedy_charge_data.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载基准解充电表", csv_greedy_charge, f"公交充电表_基准解_{current_date.strftime('%Y%m%d')}.csv")
            st.divider()
        if st.session_state.schedule_data is not None:
            st.subheader("🎯 精确解（遗传算法）排班表")
            st.dataframe(st.session_state.schedule_data, use_container_width=True)
            csv_genetic = st.session_state.schedule_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 下载精确解排班表", csv_genetic, f"公交排班表_精确解_{current_date.strftime('%Y%m%d')}.csv")
            if st.session_state.charge_data is not None:
                with st.expander("精确解充电表"):
                    st.dataframe(st.session_state.charge_data, use_container_width=True)
                    csv_genetic_charge = st.session_state.charge_data.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载精确解充电表", csv_genetic_charge, f"公交充电表_精确解_{current_date.strftime('%Y%m%d')}.csv")
        else:
            st.info("请先完成优化求解")
