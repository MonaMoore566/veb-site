from flask import Flask, render_template, request, jsonify
from GNNText import MainAI
import torch
import os
# ssh -R 80:localhost:5000 serveo.net
app = Flask(__name__)

class WebAIAssistant:
    """Класс-обертка для взаимодействия с нейросетью через веб-интерфейс"""
   
    def __init__(self, model_path="saved_model"):
        self.model_path = model_path
        self.ai = None
        self.initialized = False

    def initialize(self):
        """Инициализация модели"""
        try:
            self.ai = MainAI(model_dir=self.model_path)
            self.initialized = True
            return True, "ИИ успешно инициализирован и готов к работе!"
        except Exception as e:
            return False, f"Ошибка инициализации модели: {str(e)}"

assistant = WebAIAssistant()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if not assistant.initialized:
            return jsonify({'error': 'Модель не инициализирована'})
        
        prompt = request.form.get('prompt', '')
        max_length = int(request.form.get('max_length', 30))
        temperature = float(request.form.get('temperature', 0.6))
        top_k = int(request.form.get('top_k', 30))
        top_p = float(request.form.get('top_p', 0.60))
        
        if not prompt:
            return jsonify({'error': 'Промпт не может быть пустым'})
        
        try:
            response = assistant.ai.generate_text(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': f'Ошибка генерации: {str(e)}'})
    
    return render_template('index.html')

@app.route('/init', methods=['POST'])
def init_model():
    success, message = assistant.initialize()
    return jsonify({'success': success, 'message': message})

@app.route('/status')
def status():
    device = "Неизвестно"
    if assistant.ai and assistant.initialized:
        device = str(assistant.ai.device)
        if device == "cuda":
            device = f"GPU: {torch.cuda.get_device_name(0)}"
    return jsonify({
        'initialized': assistant.initialized,
        'device': device
    })

if __name__ == '__main__':
    # Создаем папку для шаблонов, если её нет
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Создаем HTML шаблон, если его нет
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write('''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Интерфейс для нейросети</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0 auto;
				padding: 0;
				margin: 0;
				box-sizing: border-box;
				scroll-behavior: smooth;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .chat-box {
            height: 300px;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 10px;
            overflow-y: auto;
            background-color: #fafafa;
            border-radius: 4px;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 4px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
            margin-right: 5%;
        }
        .ai-message {
            background-color: #f1f1f1;
            margin-right: 20%;
            margin-left: 5%;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            min-height: 60px;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .controls {
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }
        .control-group {
            margin-bottom: 10px;
        }
        label {
            display: inline-block;
            width: 120px;
        }
        input[type="number"] {
            width: 60px;
            padding: 5px;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
        }
        .status-error {
            background-color: #ffebee;
            color: #d32f2f;
        }
        .status-success {
            background-color: #e8f5e9;
            color: #388e3c;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Интерфейс для нейросети</h1>
        
        <div id="status" class="status"></div>
        
        <div class="chat-box" id="chat-box"></div>
        
        <textarea id="prompt" placeholder="Введите ваш запрос..."></textarea>
        
        <button id="send-btn" onclick="sendPrompt()">Отправить</button>
        
        <div class="controls">
            <h3>Параметры генерации</h3>
            <div class="control-group">
                <label for="max_length">Длина текста:</label>
                <input type="number" id="max_length" value="30" min="10" max="500">
            </div>
            <div class="control-group">
                <label for="temperature">Temperature:</label>
                <input type="number" id="temperature" value="0.6" min="0.1" max="1.5" step="0.1">
            </div>
            <div class="control-group">
                <label for="top_k">Top-K:</label>
                <input type="number" id="top_k" value="20" min="1" max="100">
            </div>
            <div class="control-group">
                <label for="top_p">Top-P:</label>
                <input type="number" id="top_p" value="0.60" min="0.1" max="1" step="0.05">
            </div>
        </div>
        
        <button id="init-btn" onclick="initModel()">Инициализировать модель</button>
    </div>

    <script>
        function updateStatus(message, isError = false) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = isError ? 'status status-error' : 'status status-success';
        }

        function addMessage(text, isUser = true) {
            const chatBox = document.getElementById('chat-box');
            const messageDiv = document.createElement('div');
            messageDiv.className = isUser ? 'message user-message' : 'message ai-message';
            messageDiv.textContent = text;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function initModel() {
            const btn = document.getElementById('init-btn');
            btn.disabled = true;
            updateStatus("Инициализация модели...");
            
            fetch('/init', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus(data.message);
                    checkStatus();
                } else {
                    updateStatus(data.message, true);
                    btn.disabled = false;
                }
            })
            .catch(error => {
                updateStatus("Ошибка при инициализации: " + error, true);
                btn.disabled = false;
            });
        }

        function sendPrompt() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;
            
            const max_length = document.getElementById('max_length').value;
            const temperature = document.getElementById('temperature').value;
            const top_k = document.getElementById('top_k').value;
            const top_p = document.getElementById('top_p').value;
            
            addMessage(prompt, true);
            document.getElementById('prompt').value = '';
            document.getElementById('send-btn').disabled = true;
            
            fetch('/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'prompt': prompt,
                    'max_length': max_length,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    addMessage("Ошибка: " + data.error, false);
                } else {
                    addMessage(data.response, false);
                }
                document.getElementById('send-btn').disabled = false;
            })
            .catch(error => {
                addMessage("Ошибка соединения: " + error, false);
                document.getElementById('send-btn').disabled = false;
            });
        }

        function checkStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                let statusText = `Статус: ${data.initialized ? 'Модель инициализирована' : 'Модель не инициализирована'}`;
                if (data.initialized) {
                    statusText += ` | Устройство: ${data.device}`;
                }
                updateStatus(statusText);
            });
        }

        // Проверяем статус при загрузке страницы
        checkStatus();
        
        // Отправка по Enter (но не Shift+Enter)
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendPrompt();
            }
        });
    </script>
</body>
</html>
            ''')
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True )
