from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from datetime import datetime

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from S002 import AgricultureAISystem
    agri_ai_system = AgricultureAISystem()
    AI_SYSTEM_LOADED = True
except Exception as e:
    print(f"❌ AI系统加载失败: {e}")
    AI_SYSTEM_LOADED = False

app = FastAPI(
    title="Kissan-Dost API",
    description="农业智能助手后端API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_sensor_data = {}
chat_history = []

@app.on_event("startup")
async def startup_event():
    print("🚀 初始化农业AI系统...")
    if AI_SYSTEM_LOADED:
        try:
            agri_ai_system.setup_iot_sensors(None)
            print("✅ 农业AI系统初始化完成")
        except Exception as e:
            print(f"❌ AI系统初始化失败: {e}")
    else:
        print("⚠️ AI系统未加载，使用降级模式")

@app.get("/")
async def root():
    return {"message": "Kissan-Dost API 服务运行中", "status": "healthy"}

@app.get("/health")
async def health_check():
    if AI_SYSTEM_LOADED:
        try:
            system_status = agri_ai_system.get_system_status()
        except:
            system_status = {"status": "ai_system_error"}
    else:
        system_status = {"status": "ai_system_not_loaded"}
    
    return {
        "status": "healthy", 
        "service": "kissan-dost-backend",
        "ai_system_status": system_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/system-status")
async def get_system_status():
    if AI_SYSTEM_LOADED:
        try:
            return agri_ai_system.get_system_status()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "ai_system_not_loaded"}

@app.post("/api/v1/ingest")
async def ingest_sensor_data(data: dict):
    global latest_sensor_data
    try:
        latest_sensor_data = data
        print(f"📊 收到传感器数据: {data.get('sensor_id', 'unknown')} - {data.get('timestamp', 'unknown')}")
        return {
            "status": "success", 
            "message": "数据接收成功",
            "data_received": {
                "sensor_id": data.get("sensor_id"),
                "location": data.get("location"),
                "timestamp": data.get("timestamp")
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"数据处理失败: {str(e)}"}

@app.post("/api/v1/chat")
async def chat_endpoint(request: dict):
    global chat_history, latest_sensor_data
    try:
        user_id = request.get("user_id", "unknown")
        user_message = request.get("message", "")
        location = request.get("location", "field_3")
        language = request.get("language", "zh-CN")
        
        print(f"💬 收到用户消息: {user_message}")
        
        sensor_data_for_ai = {}
        if latest_sensor_data and 'readings' in latest_sensor_data:
            sensor_data_for_ai = latest_sensor_data['readings']
            if 'npk' in sensor_data_for_ai and isinstance(sensor_data_for_ai['npk'], dict):
                npk_data = sensor_data_for_ai.pop('npk')
                sensor_data_for_ai.update({
                    'npk_nitrogen': npk_data.get('nitrogen', 0),
                    'npk_phosphorus': npk_data.get('phosphorus', 0),
                    'npk_potassium': npk_data.get('potassium', 0)
                })
        
        if AI_SYSTEM_LOADED:
            model_a_output = agri_ai_system.model_a.predict(sensor_data_for_ai)
            ai_advice = agri_ai_system.model_b.predict(
                model_a_output, 
                sensor_data_for_ai, 
                user_message=user_message
            )
        else:
            ai_advice = generate_fallback_response(user_message, sensor_data_for_ai)
        
        response_data = {
            "response": ai_advice,
            "advice": "请参考上述建议",
            "confidence": 0.85,
            "data_sources": {
                "sensor_data": latest_sensor_data.get('readings', {}),
                "weather": "未来24小时无雨",
                "market": "柑橘价格稳定",
                "ai_system": "农业AI分析系统"
            },
            "actions": [{"type": "general", "description": "遵循AI建议", "urgency": "medium"}],
            "status": "success"
        }
        
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_message": user_message,
            "ai_response": ai_advice,
            "location": location
        }
        chat_history.append(chat_entry)
        
        if len(chat_history) > 100:
            chat_history = chat_history[-100:]
        
        return response_data
        
    except Exception as e:
        print(f"❌ 聊天处理错误: {e}")
        return {
            "response": "抱歉，系统暂时无法处理您的请求。",
            "advice": "请稍后重试",
            "confidence": 0.0,
            "status": "error",
            "error": str(e)
        }

def generate_fallback_response(user_message, sensor_data):
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in ['你好', '您好', 'hello']):
        return "🌱 您好！我是果农助手，可以为您提供柑橘种植建议。"
    
    if any(word in message_lower for word in ['浇水', '灌溉']):
        moisture = sensor_data.get('soil_moisture', 50)
        return f"💧 当前土壤湿度{moisture}%，建议{'立即浇水' if moisture < 30 else '保持当前灌溉'}"
    
    if any(word in message_lower for word in ['施肥', '肥料']):
        return "🌿 建议使用NPK复合肥，春季追氮肥，夏季增施磷钾肥"
    
    return "🤔 我可以帮您分析土壤湿度、施肥、病虫害等问题，请具体说明您想了解的内容。"

@app.get("/api/v1/chat-history")
async def get_chat_history(limit: int = 10):
    return {
        "status": "success",
        "history": chat_history[-limit:] if chat_history else []
    }

@app.get("/api/v1/sensor-data")
async def get_sensor_data():
    return {
        "status": "success",
        "sensor_data": latest_sensor_data,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/analyze")
async def analyze_farm():
    if not AI_SYSTEM_LOADED:
        return {"status": "error", "message": "AI系统未加载"}
    
    try:
        advice = agri_ai_system.inference_pipeline()
        return {
            "status": "success",
            "analysis": advice,
            "system_status": agri_ai_system.get_system_status(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": f"分析失败: {str(e)}"}

if __name__ == "__main__":
    print("🚀 启动Kissan-Dost后端服务...")
    print(f"📂 工作目录: {os.getcwd()}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)