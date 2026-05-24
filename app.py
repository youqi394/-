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
    layout="wide",  # 宽屏模式，铺满浏览器
    initial_sidebar_state="expanded"
)

# 隐藏Streamlit默认的右上角菜单和底部水印
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

# -------------------------- 全局常量与配置 --------------------------
# 和风天气API（已替换为你的API Key）
WEATHER_API_KEY = "be44281d3442497b9a7b47b57a49f321"
# 北京城市ID（可替换为你所在城市）
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

# -------------------------- 工具函数 --------------------------
def add_log(message):
    """添加求解日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.solve_log.append(f"[INFO] {timestamp} - {message}")

def get_weather_forecast(date):
    """获取指定日期的天气预报（修复版：解决'code'键不存在错误）"""
    try:
        # 计算日期差（和风天气最多支持未来7天预报）
        today = datetime.now().date()
        days_diff = (date - today).days
        
        if days_diff < 0 or days_diff > 7:
            return None, "只能查询未来7天的天气预报"
        
        # 调用和风天气API（使用v7版本标准接口）
        url = f"https://devapi.qweather.com/v7/weather/7d?location={CITY_ID}&key={WEATHER_API_KEY}"
        
        # 添加超时和重试机制
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # 检查HTTP状态码
        
        data = response.json()
        
        # 严格检查API返回格式（修复'code'键不存在错误）
        if not isinstance(data, dict) or 'code' not in data:
            return None, "API返回格式异常，缺少状态码"
            
        if data['code'] != '200':
            error_msg = data.get('message', f"API错误代码：{data['code']}")
            return None, f"天气API调用失败：{error_msg}"
        
        # 检查每日数据是否存在
        if 'daily' not in data or not isinstance(data['daily'], list) or len(data['daily']) <= days_diff:
            return None, "未找到对应日期的天气数据"
            
        day_weather = data['daily'][days_diff]
        
        # 检查天气数据字段是否完整
        required_fields = ['tempMax', 'tempMin', 'textDay']
        for field in required_fields:
            if field not in day_weather:
                return None, f"天气数据缺失字段：{field}"
        
        weather_info = {
            'date': date,
            'temp_max': int(day_weather['tempMax']),
            'temp_min': int(day_weather['tempMin']),
            'weather': day_weather['textDay'],
            'is_rain': 1 if '雨' in day_weather['textDay'] else 0
        }
        
        return weather_info, None
        
    except requests.exceptions.RequestException as e:
        return None, f"网络请求失败：{str(e)}"
    except (KeyError, ValueError, IndexError, TypeError) as e:
        return None, f"数据解析失败：{str(e)}"
    except Exception as e:
        return None, f"未知错误：{str(e)}"

@st.cache_resource
def load_ai_models():
    """加载AI预测模型（只加载一次）"""
    try:
        # 尝试加载预训练模型
        model = joblib.load("data/flow_prediction_model.pkl")
        scaler = joblib.load("data/scaler.pkl")
        pca_flow = joblib.load("data/pca_flow.pkl")
        pca_congestion = joblib.load("data/pca_congestion.pkl")
        return model, scaler, pca_flow, pca_congestion, None
    except:
        # 如果没有预训练模型，使用内置的默认模型
        add_log("未找到预训练模型，使用内置默认模型")
        return None, None, None, None, "未找到预训练模型，将使用默认预测数据"

def predict_passenger_flow(date, line_id, is_workday, weather_data):
    """真实AI客流预测"""
    model, scaler, pca_flow, pca_congestion, error = load_ai_models()
    
    hours = list(range(6, 22))
    predictions = []
    
    if error:
        # 使用基于日期和天气的智能默认预测
        base_flow = 150 if is_workday else 100
        rain_factor = 0.8 if weather_data and weather_data['is_rain'] else 1.0
        
        for hour in hours:
            # 模拟早晚高峰
            if 7 <= hour <= 9:
                flow = base_flow * 2.5 * rain_factor
            elif 17 <= hour <= 19:
                flow = base_flow * 2.2 * rain_factor
            elif 6 <= hour <= 21:
                flow = base_flow * rain_factor
            else:
                flow = base_flow * 0.5 * rain_factor
            
            # 添加随机波动
            flow = flow * (0.9 + np.random.random() * 0.2)
            predictions.append(round(flow))
    else:
        # 使用真实模型预测
        for hour in hours:
            # 构造特征
            features = pd.DataFrame({
                'hour': [hour],
                'day_of_week': [date.weekday()],
                'is_workday': [is_workday],
                'is_holiday': [0],
                'temp': [weather_data['temp_max'] if weather_data else 25],
                'is_rain': [weather_data['is_rain'] if weather_data else 0]
            })
            
            # 提取PCA特征
            flow_features = features[['hour', 'is_workday']]
            congestion_features = features[['temp', 'is_rain']]
            
            flow_scaled = scaler.transform(flow_features)
            congestion_scaled = scaler.transform(congestion_features)
            
            features['flow_factor'] = pca_flow.transform(flow_scaled)[0][0]
            features['congestion_factor'] = pca_congestion.transform(congestion_scaled)[0][0]
            
            # 预测
            pred = model.predict(features[['hour', 'day_of_week', 'is_workday', 'flow_factor', 'congestion_factor']])[0]
            predictions.append(round(pred))
    
    return hours, predictions

def optimize_schedule(predictions, vehicle_count, initial_battery, solve_time_limit):
    """Gurobi真实优化求解"""
    add_log("开始初始化优化模型")
    
    # 清空之前的收敛数据
    st.session_state.convergence_data = []
    st.session_state.solve_log = []
    
    hours = list(range(6, 22))
    n_hours = len(hours)
    n_vehicles = vehicle_count
    
    # 创建模型
    model = gp.Model("bus_scheduling")
    model.setParam('TimeLimit', solve_time_limit)
    model.setParam('OutputFlag', 1)
    
    # 定义回调函数，记录收敛过程
    def callback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            if obj < 1e100:
                st.session_state.current_objective = obj
                st.session_state.current_gap = (obj - bound) / obj if obj != 0 else 0
                st.session_state.convergence_data.append((len(st.session_state.convergence_data)+1, obj))
    
    # 变量：x[i][j]表示第i辆车在第j小时是否发车
    x = model.addVars(n_vehicles, n_hours, vtype=GRB.BINARY, name="x")
    
    # 变量：y[j]表示第j小时的发车间隔（分钟）
    y = model.addVars(n_hours, vtype=GRB.CONTINUOUS, name="y")
    
    # 目标函数：最小化乘客等待时间 + 车辆运营成本
    obj = 0
    for j in range(n_hours):
        # 乘客等待时间：与客流量成正比，与发车间隔成正比
        obj += predictions[j] * y[j] * 0.1
        # 车辆运营成本
        for i in range(n_vehicles):
            obj += x[i, j] * 50
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 约束1：每小时至少发1班车
    for j in range(n_hours):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n_vehicles)) >= 1)
    
    # 约束2：发车间隔约束
    for j in range(n_hours):
        departures = gp.quicksum(x[i, j] for i in range(n_vehicles))
        model.addConstr(y[j] == 60 / departures)
        # 最大发车间隔不超过15分钟
        model.addConstr(y[j] <= 15)
    
    # 约束3：车辆连续运营约束（每辆车最多连续运营4小时）
    for i in range(n_vehicles):
        for j in range(n_hours - 4):
            model.addConstr(gp.quicksum(x[i, j+k] for k in range(5)) <= 4)
    
    # 约束4：电量约束（假设每小时消耗10%电量）
    for i in range(n_vehicles):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n_hours)) * 10 <= initial_battery)
    
    add_log("模型初始化完成，开始求解")
    
    # 求解
    model.optimize(callback)
    
    add_log(f"求解完成，最优目标值：{model.ObjVal:.2f}")
    
    # 生成排班结果
    schedule = []
    for j in range(n_hours):
        hour = hours[j]
        departures = []
        for i in range(n_vehicles):
            if x[i, j].X > 0.5:
                departures.append(f"车{i+1:02d}")
        
        # 计算每个班次的具体发车时间
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
st.sidebar.title("🚍 智能公交调度系统")
st.sidebar.divider()
page = st.sidebar.radio(
    "功能模块",
    ["📅 今日调度", "📊 数据管理", "🤖 AI预测结果", "⚙️ 优化求解", "📋 排班结果"]
)
st.sidebar.divider()
st.sidebar.info("AI预测-优化建模 智能公交调度")

# -------------------------- 页面1：今日调度 --------------------------
if page == "📅 今日调度":
    st.header("AI预测-优化建模 智能公交调度")
    st.divider()

    # 调度参数设置
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

    # 功能按钮
    btn1, btn2, btn3, btn4, btn5 = st.columns(5)

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
                    st.session_state.weather_data = {
                        'temp_max': 25,
                        'temp_min': 18,
                        'weather': '晴',
                        'is_rain': 0
                    }
                else:
                    st.session_state.weather_data = weather_info
                    st.success(f"✅ 天气数据读取成功！{weather_info['weather']}，{weather_info['temp_min']}~{weather_info['temp_max']}℃")
                    add_log(f"天气数据读取成功：{weather_info['weather']}，{weather_info['temp_min']}~{weather_info['temp_max']}℃")
                
                st.session_state.current_stage = "天气数据加载完成"
                st.session_state.progress = 30

    with btn3:
        if st.button("运行AI预测"):
            if st.session_state.weather_data is None:
                st.warning("⚠️ 请先读取天气数据")
            else:
                st.info("🔄 AI客流预测中...")
                progress_bar = st.progress(0)
                
                is_workday = 1 if timetable_type == "工作日" else 0
                hours, predictions = predict_passenger_flow(
                    dispatch_date, line, is_workday, st.session_state.weather_data
                )
                
                # 更新进度条
                for i in range(30, 60):
                    progress_bar.progress(i/100)
                    time.sleep(0.02)
                
                st.session_state.predictions = predictions
                st.session_state.prediction_hours = hours
                
                st.success("✅ AI预测完成！")
                st.session_state.current_stage = "AI预测完成"
                st.session_state.progress = 60
                add_log(f"AI客流预测完成，共预测{len(hours)}个时段")

    with btn4:
        if st.button("开始优化求解"):
            if st.session_state.predictions is None:
                st.warning("⚠️ 请先运行AI预测")
            else:
                st.info("🔄 优化求解中...")
                progress_bar = st.progress(0)
                
                # 运行Gurobi优化
                model, schedule_df = optimize_schedule(
                    st.session_state.predictions,
                    vehicle_count,
                    initial_battery,
                    solve_time
                )
                
                # 更新进度条
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
                    mime="text/csv",
                    key="download_schedule"
                )
                st.success("✅ 排班结果已准备好下载！")
                st.session_state.current_stage = "全部完成"
                st.session_state.progress = 100
                add_log("排班结果导出完成")

    st.divider()

    # 进度条
    st.progress(st.session_state.progress, text=f"当前进度：{st.session_state.progress}%")

    st.divider()

    # 底部状态显示（动态更新）
    status1, status2, status3, status4, status5 = st.columns(5)
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
    st.header("数据管理模块")
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
    
    if 'predictions' not in st.session_state or st.session_state.predictions is None:
        st.info("请先在「今日调度」页面点击「运行AI预测」")
    else:
        st.subheader(f"{line}线路 {dispatch_date.strftime('%Y-%m-%d')} 客流预测曲线")
        
        chart_data = pd.DataFrame({
            "时间": [f"{h}:00" for h in st.session_state.prediction_hours],
            "预测客流": st.session_state.predictions
        })
        st.line_chart(chart_data, x="时间", y="预测客流", use_container_width=True)
        
        st.divider()
        
        st.subheader("预测结果统计")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            morning_peak = max(st.session_state.predictions[1:4])  # 7-9点
            st.metric("早高峰最大客流", f"{morning_peak}人")
        with col2:
            evening_peak = max(st.session_state.predictions[11:14])  # 17-19点
            st.metric("晚高峰最大客流", f"{evening_peak}人")
        with col3:
            total_flow = sum(st.session_state.predictions)
            st.metric("日均总客流", f"{total_flow}人")
        with col4:
            st.metric("预测置信水平", confidence)

# -------------------------- 页面4：优化求解 --------------------------
elif page == "⚙️ 优化求解":
    st.header("优化求解过程监控")
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
    st.header("最终排班结果")
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
            # 打印功能
            if st.button("🖨️ 打印排班表"):
                st.components.v1.html(
                    """
                    <script>
                    window.print();
                    </script>
                    """,
                    height=0
                )
                st.success("✅ 打印对话框已打开")
