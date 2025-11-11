from S000 import *
import random
from datetime import datetime
import requests

class IoTDataCollector:
    def __init__(self):
        self.sensors = {}
        self.data_buffer = []
        self.backend_url = "http://localhost:8000"
    
    def add_sensor(self, sensor_type, sensor_id, config):
        self.sensors[sensor_id] = {
            'type': sensor_type,
            'config': config,
            'last_reading': None
        }
        printLog(f"添加传感器: {sensor_id} ({sensor_type})")
    
    def collect_data(self):
        sensor_data = {}
        for sensor_id, sensor_info in self.sensors.items():
            sensor_type = sensor_info['type']
            if sensor_type == 'soil_moisture':
                reading = round(random.uniform(20, 60), 1)
            elif sensor_type == 'temperature':
                reading = round(random.uniform(15, 35), 1)
            elif sensor_type == 'humidity':
                reading = round(random.uniform(40, 90), 1)
            elif sensor_type == 'ph_sensor':
                reading = round(random.uniform(5.0, 7.5), 1)
            elif sensor_type == 'npk_sensor':
                reading = {
                    'nitrogen': random.randint(30, 70),
                    'phosphorus': random.randint(20, 60),
                    'potassium': random.randint(25, 65)
                }
            else:
                reading = random.uniform(0, 100)
            sensor_data[sensor_id] = reading
            self.sensors[sensor_id]['last_reading'] = reading
        return sensor_data
    
    def preprocess_data(self, raw_data):
        processed = {}
        for sensor_id, reading in raw_data.items():
            if isinstance(reading, (int, float)):
                if 0 <= reading <= 100:
                    processed[sensor_id] = reading
                else:
                    printLog(f"传感器 {sensor_id} 数据异常: {reading}", "WARNING")
            elif isinstance(reading, dict):
                processed[sensor_id] = reading
            else:
                printLog(f"传感器 {sensor_id} 数据格式错误", "WARNING")
        return processed
    
    def send_to_backend(self, data):
        try:
            formatted_data = {
                "sensor_id": "agri_sensor_001",
                "location": "field_3",
                "timestamp": datetime.now().isoformat(),
                "readings": data,
                "metadata": {"crop_type": "citrus", "growth_stage": "flowering"}
            }
            response = requests.post(
                f"{self.backend_url}/api/v1/ingest",
                json=formatted_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code == 200:
                printLog(f"数据发送成功: {len(data)}个传感器读数")
                return True
            else:
                printLog(f"数据发送失败: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            printLog(f"发送数据时出错: {e}", "ERROR")
            return False

class AgricultureAIModel(BaseModel):
    def __init__(self, model_name, model_type):
        super().__init__(model_name)
        self.model_type = model_type
        self.feature_columns = []
        self.target_column = ""
        self.training_history = []
    
    def feature_engineering(self, data):
        try:
            if isinstance(data, dict):
                features = {}
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        features[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            features[f"{key}_{sub_key}"] = sub_value
                return features
            else:
                printLog("特征工程: 输入数据格式不支持", "WARNING")
                return data
        except Exception as e:
            printLog(f"特征工程出错: {e}", "ERROR")
            return data
    
    def log_training(self, epoch, loss, accuracy=None):
        log_entry = {
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        self.training_history.append(log_entry)

class SensorDataModel(AgricultureAIModel):
    def __init__(self):
        super().__init__("sensor_data_model", "regression")
        self.feature_columns = [
            'temperature', 'humidity', 'soil_moisture', 
            'soil_ph', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium'
        ]
        self.target_column = "crop_health_index"
    
    def train(self, train_data, **kwargs):
        try:
            printLog("开始训练传感器数据模型...")
            self.model = "simulated_sensor_model"
            printLog("模拟传感器模型训练完成")
        except Exception as e:
            printLog(f"模型训练失败: {e}", "ERROR")
            self.model = "fallback_sensor_model"
    
    def predict(self, input_data, **kwargs):
        try:
            processed_data = self.preprocess_sensor_data(input_data)
            if self.model == "simulated_sensor_model" or self.model == "fallback_sensor_model":
                moisture = processed_data.get('soil_moisture', 50)
                if moisture < 30:
                    return "needs_water"
                elif moisture > 60:
                    return "too_much_water"
                else:
                    return "healthy"
            else:
                prediction = self.model.predict([list(processed_data.values())])[0]
                return self.interpret_prediction(prediction)
        except Exception as e:
            printLog(f"预测出错: {e}", "ERROR")
            return "unknown"
    
    def preprocess_sensor_data(self, raw_data):
        processed = {}
        try:
            for key, value in raw_data.items():
                if isinstance(value, (int, float)):
                    processed[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        norm_key = f"npk_{sub_key}"
                        processed[norm_key] = sub_value
            return processed
        except Exception as e:
            printLog(f"数据预处理出错: {e}", "ERROR")
            return {feature: 50 for feature in self.feature_columns}
    
    def interpret_prediction(self, prediction_value):
        if prediction_value < 0.3:
            return "needs_water"
        elif prediction_value < 0.5:
            return "needs_nutrients"
        elif prediction_value < 0.7:
            return "healthy"
        else:
            return "excellent"

class LanguageTranslationModel(AgricultureAIModel):
    def __init__(self):
        super().__init__("agriculture_language_model", "translation")
        self.agriculture_knowledge_base = {}
        self.language_templates = {}
        self.user_context = {}
        self.load_agriculture_templates()
        self.build_agriculture_knowledge_base()
    
    def train(self, train_data, **kwargs):
        printLog("开始训练农业语言翻译模型...")
        try:
            self.model = "simulated_language_model"
            printLog("农业语言翻译模型训练完成")
        except Exception as e:
            printLog(f"语言模型训练失败: {e}", "ERROR")
            self.model = "fallback_language_model"
    
    def predict(self, model_a_output, sensor_data=None, user_message=None, **kwargs):
        try:
            if user_message:
                return self.generate_contextual_response(user_message, model_a_output, sensor_data)
            else:
                return self.generate_detailed_advice(model_a_output, sensor_data)
        except Exception as e:
            printLog(f"语言翻译出错: {e}", "ERROR")
            return "目前无法提供农业建议，请稍后重试。"
    
    def generate_contextual_response(self, user_message, crop_status, sensor_data):
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['你好', '您好', 'hello', 'hi', '嗨']):
            return "🌱 您好！我是果农助手，专门为柑橘种植提供智能建议。请问您想了解什么？"
        
        if any(word in message_lower for word in ['谢谢', '感谢', '多谢']):
            return "🙏 不客气！随时为您提供农业咨询服务。"
        
        if any(word in message_lower for word in ['浇水', '灌溉', '水分', '湿度']):
            return self.generate_water_advice(crop_status, sensor_data)
        
        if any(word in message_lower for word in ['施肥', '肥料', '营养', 'npk']):
            return self.generate_fertilizer_advice(crop_status, sensor_data)
        
        if any(word in message_lower for word in ['病虫害', '虫害', '病害', '防治']):
            return self.generate_pest_control_advice()
        
        if any(word in message_lower for word in ['温度', '气温', '天气']):
            return self.generate_temperature_advice(sensor_data)
        
        if any(word in message_lower for word in ['土壤', 'ph', '酸碱']):
            return self.generate_soil_advice(sensor_data)
        
        if any(word in message_lower for word in ['怎么样', '情况', '状态', '如何']):
            return self.generate_detailed_advice(crop_status, sensor_data)
        
        return f"🤔 您问的是 '{user_message}' 吗？我可以帮您分析：\n\n" + \
               self.generate_detailed_advice(crop_status, sensor_data)
    
    def generate_water_advice(self, crop_status, sensor_data):
        moisture = sensor_data.get('soil_moisture', 50)
        
        if moisture < 25:
            advice = f"💧 **急需浇水**\n当前土壤湿度只有{moisture}%，严重不足！\n建议立即灌溉，浇水量为每亩10-15立方米。"
        elif moisture < 35:
            advice = f"💧 **需要浇水**\n当前土壤湿度{moisture}%偏低。\n建议今天内安排灌溉，浇水量为每亩8-12立方米。"
        elif moisture > 65:
            advice = f"⚠️ **水分过多**\n当前土壤湿度{moisture}%过高。\n建议暂停浇水，注意排水防涝。"
        else:
            advice = f"✅ **水分适宜**\n当前土壤湿度{moisture}%处于理想范围。\n保持当前灌溉频率即可。"
        
        advice += "\n\n🌱 **柑橘浇水知识**: 开花期保持30-40%湿度，果实膨大期保持40-50%湿度。"
        return advice
    
    def generate_fertilizer_advice(self, crop_status, sensor_data):
        nitrogen = sensor_data.get('npk_nitrogen', 50)
        phosphorus = sensor_data.get('npk_phosphorus', 40)
        potassium = sensor_data.get('npk_potassium', 45)
        
        advice = f"🌿 **当前营养状况**:\n"
        advice += f"• 氮(N): {nitrogen}% {'✅充足' if nitrogen > 40 else '⚠️不足'}\n"
        advice += f"• 磷(P): {phosphorus}% {'✅充足' if phosphorus > 30 else '⚠️不足'}\n"
        advice += f"• 钾(K): {potassium}% {'✅充足' if potassium > 35 else '⚠️不足'}\n\n"
        
        if nitrogen < 40 or phosphorus < 30 or potassium < 35:
            advice += "💡 **施肥建议**:\n"
            if nitrogen < 40:
                advice += "• 补充氮肥促进新梢生长\n"
            if phosphorus < 30:
                advice += "• 补充磷肥促进根系发育\n"
            if potassium < 35:
                advice += "• 补充钾肥提高果实品质\n"
            advice += "\n推荐NPK复合肥，比例2:1:1"
        else:
            advice += "✅ **营养状况良好**，保持当前施肥方案即可。"
        
        return advice
    
    def generate_pest_control_advice(self):
        advice = "🐛 **柑橘常见病虫害防治**:\n\n"
        advice += "• **红蜘蛛**: 使用阿维菌素或螺螨酯喷雾\n"
        advice += "• **蚜虫**: 使用吡虫啉或啶虫脒防治\n"
        advice += "• **炭疽病**: 使用咪鲜胺或苯醚甲环唑\n"
        advice += "• **溃疡病**: 使用氢氧化铜或春雷霉素\n\n"
        advice += "💡 **预防措施**:\n"
        advice += "• 保持果园通风透光\n"
        advice += "• 及时清理落叶病果\n"
        advice += "• 合理修剪增强树势"
        
        return advice
    
    def generate_temperature_advice(self, sensor_data):
        temperature = sensor_data.get('temperature', 25)
        
        if temperature < 10:
            advice = f"❄️ **温度过低**\n当前温度{temperature}℃，柑橘可能受冻害。\n建议采取保温措施。"
        elif temperature < 15:
            advice = f"🌡️ **温度偏低**\n当前温度{temperature}℃，生长缓慢。\n注意观察植株状态。"
        elif temperature > 35:
            advice = f"🔥 **温度过高**\n当前温度{temperature}℃，可能造成日灼。\n建议适当遮阴。"
        else:
            advice = f"✅ **温度适宜**\n当前温度{temperature}℃是柑橘生长的理想温度。"
        
        advice += "\n\n🌡️ **适宜温度**: 柑橘生长最适温度为15-30℃。"
        return advice
    
    def generate_soil_advice(self, sensor_data):
        ph = sensor_data.get('soil_ph', 6.5)
        
        if ph < 5.5:
            advice = f"🧪 **土壤过酸**\n当前pH值{ph}，需要改良。\n建议施用石灰调节。"
        elif ph > 7.5:
            advice = f"🧪 **土壤过碱**\n当前pH值{ph}，需要改良。\n建议施用硫磺或有机肥。"
        else:
            advice = f"✅ **土壤酸碱度适宜**\n当前pH值{ph}是柑橘生长的理想范围。"
        
        advice += "\n\n🌱 **适宜pH**: 柑橘适宜土壤pH为5.5-7.5。"
        return advice
    
    def generate_detailed_advice(self, crop_status, sensor_data):
        base_advice = self.translate_to_natural_language(crop_status)
        details = []
        
        moisture = sensor_data.get('soil_moisture')
        if moisture is not None:
            if moisture < 25:
                details.append(f"土壤湿度{moisture}%严重不足，急需灌溉")
            elif moisture < 35:
                details.append(f"土壤湿度{moisture}%偏低，需要浇水")
            elif moisture > 65:
                details.append(f"土壤湿度{moisture}%过高，注意排水")
            else:
                details.append(f"土壤湿度{moisture}%适宜")
        
        temperature = sensor_data.get('temperature')
        if temperature is not None:
            if temperature < 10:
                details.append(f"温度{temperature}℃过低，注意防冻")
            elif temperature > 35:
                details.append(f"温度{temperature}℃过高，注意遮阴")
            else:
                details.append(f"温度{temperature}℃适宜")
        
        if details:
            advice = base_advice + "\n\n📊 **详细分析**:\n• " + "\n• ".join(details)
        else:
            advice = base_advice + "\n\n💡 建议定期检查土壤湿度和营养状况。"
        
        return advice
    
    def translate_to_natural_language(self, ai_output):
        templates = {
            'healthy': "🌱 **作物生长状况良好**\n各项指标正常，继续保持当前管理措施。",
            'needs_water': "💧 **需要灌溉**\n土壤湿度偏低，建议及时浇水。",
            'needs_nutrients': "🌿 **需要施肥**\n检测到营养不足，建议适量补充肥料。",
            'too_much_water': "⚠️ **水分过多**\n土壤湿度过高，建议减少灌溉并改善排水。",
            'pest_risk': "🐛 **病虫害风险**\n环境条件适宜病虫害发生，建议加强预防。",
            'excellent': "🎉 **生长状况极佳**\n继续保持优良的管理措施！",
            'unknown': "❓ **状态未知**\n建议人工检查作物生长情况。"
        }
        return templates.get(ai_output, "状态未知，建议人工检查")
    
    def build_agriculture_knowledge_base(self):
        self.agriculture_knowledge_base = {
            'citrus': {
                'irrigation': '柑橘在开花期需要保持土壤湿度30-40%，果实膨大期需要40-50%',
                'fertilization': '春季追施氮肥，夏季增施磷钾肥，NPK比例建议2:1:1',
                'pest_control': '注意防治红蜘蛛、蚜虫，保持果园通风透光',
                'pruning': '冬季修剪弱枝病枝，夏季修剪徒长枝',
                'harvest': '果实着色均匀，可溶性固形物达到12%以上即可采收'
            }
        }
    
    def load_agriculture_templates(self):
        self.language_templates = {
            'zh-CN': {
                'greeting': '您好！我是果农助手，可以为您提供柑橘种植建议。',
                'help': '您可以问我关于土壤湿度、施肥、病虫害防治等问题。',
                'error': '抱歉，我暂时无法回答这个问题。请尝试询问其他农业相关问题。'
            }
        }