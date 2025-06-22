from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, r2_score
from textwrap import wrap
import evaluate
import subprocess
import time
import numpy as np
import os
import torch
import math
# model_name = "sberbank-ai/rugpt3small_based_on_gpt2"  # 35M параметров
# nvidia-smi, tensorboard --logdir C:\Users\Danil\Desktop\modules tensorboard --logdir=runs --reload_multifile=true
# model_name = "ai-forever/rugpt3large_based_on_gpt2"  # 760M параметров
torch.cuda.empty_cache()
class CustomCallback(TrainerCallback):
	def __init__(self, writer):
		self.writer = writer
		self.start_time = time.time()
		self.gpu_available = torch.cuda.is_available()
		# Инициализация для мониторинга GPU (если доступно)
		if self.gpu_available:
			try:
				import pynvml
				pynvml.nvmlInit()
				self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
				self.use_nvml = True
			except:
				self.use_nvml = False
	
	def on_step_end(self, args, state, control, **kwargs):
		def get_gpu_memory():
			result = subprocess.run(
				['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
				capture_output=True, 
				text=True
			)
			return float(result.stdout.strip()) / 1024  # Конвертируем МБ в ГБ
		if state.is_world_process_zero and state.global_step % 50 == 0:
			# Мониторинг GPU
			if self.gpu_available:
				try:
					# Использование памяти
					nvidia_mem = get_gpu_memory()
					self.writer.add_scalar('system/nvidia_smi_memory', nvidia_mem, state.global_step)
					mem_used = torch.cuda.memory_allocated() / 1024**3
					self.writer.add_scalar('system/gpu_mem_used', mem_used, state.global_step)
					# Загрузка GPU (разные способы в зависимости от доступности)
					if self.use_nvml:
						import pynvml
						util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
						self.writer.add_scalar('system/gpu_util', util.gpu, state.global_step)
					else:
						# Альтернативный способ оценки загрузки
						self.writer.add_scalar('system/gpu_util', min(90, mem_used * 10), state.global_step)
				except Exception as e:
					print(f"Ошибка мониторинга GPU: {str(e)}")

				current_time = time.time()
				self.writer.add_scalar('time/step_time', current_time - self.start_time, state.global_step)
				self.writer.flush()

	def on_log(self, args, state, control, logs=None, **kwargs):
		if state.is_world_process_zero:
			# Стандартные метрики
			if 'loss' in logs:
				self.writer.add_scalar('train/loss', logs['loss'], state.global_step)
			if 'eval_loss' in logs:
				self.writer.add_scalar('val/loss', logs['eval_loss'], state.global_step)

			# Learning rate
			if 'learning_rate' in logs:
				self.writer.add_scalar('meta/lr', logs['learning_rate'], state.global_step)



class MainAI():
	def __init__(self, model_name = "sberbank-ai/rugpt3small_based_on_gpt2", model_dir = None):
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Определение устройство для вычислений (GPU/CPU)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cuda")

		# Проверка поддержки mixed precision (fp16)
		self.fp16_available = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

		if model_dir and os.path.exists(model_dir):
			self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir, pad_token="[PAD]")
			self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device)
		else:
			self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="[PAD]")
			self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

		if str(self.device) == "cuda": # Должно вывести 'cuda' если GPU доступен
			print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
			print(f"Всего GPU памяти: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

		self.tokenizer.add_special_tokens({'eos_token': '</s>'})
		self.model.resize_token_embeddings(len(self.tokenizer))


	def load_and_preparation_data(self, file_path, val_size):
		"""Загрузка и подготовка данных для обучения"""
		# Загрузка данных
		try:
			if not os.path.exists(file_path):
				raise FileNotFoundError(f"Training file not found: {file_path}")
			dataset = load_dataset(path = "text", data_files ={"train": file_path})

			
			# Токенизация данных
			def tokenize_dataset(examples):
				return self.tokenizer(
					examples["text"], # данные из входных данных
					max_length = 512,
					truncation = True, # обрезание текста, если он превышает max_length
					padding=False, # Отключаем padding на этом этапе
					add_special_tokens=True,
				)
			tokenized_dataset = dataset.map(
				tokenize_dataset, # функция, каждому примеру/ батчу
				batched = True, # обработка данных батчами
				remove_columns = ["text"], # удаление столбца с текстом
				num_proc = 4 #параллельная обработка 
			)
			filtered_dataset = tokenized_dataset.filter(
				lambda x: len(x["input_ids"]) > 0,
				num_proc = 4
			)
			# разделение датасета на обучающую и валидационную части
			return filtered_dataset['train'].train_test_split(test_size = val_size, shuffle=True)
		
		except Exception as e:
			print(f"Data loading error: {e}")
			raise

	def setup_trainer(self, dataset, output_dir, writer):
		"""Настройка обучения с улучшениями"""
		print("начало обучения")
		training_args = TrainingArguments(
			output_dir = output_dir, # сохранения результатов
			overwrite_output_dir = True, # перезапись существующих файлов
			num_train_epochs = 7, # КОЛ-ВО ЭПОХ
			per_device_train_batch_size = 8, # РАЗМЕР БАТЧА ДЛЯ ТРЕНИРОВКИ
			per_device_eval_batch_size = 4, # размер батча для валидации
			eval_strategy = "steps", # стратегия оценки по шагам
			save_strategy = "steps",
			dataloader_pin_memory = True, # отключение фиксированной памяти
			dataloader_num_workers = 2,
			# gradient_checkpointing = True, # Использовать только ночью
			eval_steps = 15000, # частота оценки (каждые 500 шагов)
			save_steps = 15000, # частота сохранения модели
			save_total_limit = 2, # макс. кол-во сохраняемых моделей
			logging_steps = 50, # частота логирования (каждые 100 шагов)
			logging_dir="runs",
			learning_rate = 2e-5, # СКОРОСТЬ ОБУЧЕНИЯ 5е-5
			max_grad_norm = 1.0, # Предотвращает "взрывающиеся" градиенты
			lr_scheduler_type="cosine_with_restarts", # Плавное снижение + "перезапуски"
			weight_decay = 0.05, # РЕГУЛЯРИЗАЦИЯ ПЕРЕОБУЧЕНИЯ (добавление штрафа)
			# dropout=0.2, # ОЧЕНЬ ВАЖНО, ЕСЛИ МОДЕЛЬ НЕ УЧИТЬСЯ ПОПРОБОВАТЬ СДЕЛАТЬ!!!
			# warmup_steps = 500, # кол-во шагов перед увиличением скорости
			warmup_ratio=0.05,
			gradient_accumulation_steps = 2, # накопление градиентов перед обновлением весов
			fp16 = torch.cuda.is_available(),# использовать mixed precision, если доступно CUDA
			optim = "adamw_torch", # оптимизатор AdamW
			report_to = ["tensorboard"], # отчет в tensorboard
			load_best_model_at_end = True, # загрузка лучшей модели по окончанию обучения
			metric_for_best_model = "eval_loss", # метрика для выбора лучшей модели
			greater_is_better = False, # лучшее меньшее значение метрики
			push_to_hub = False
		)

		# Подготовка батчей данных для модели
		data_collator = DataCollatorForLanguageModeling(
			tokenizer = self.tokenizer,
			mlm = False,
			pad_to_multiple_of = 8 # Улучшение производительности на GPU
		) 
		self.model.config.dropout = 0.2
		return Trainer(
			model = self.model, # модель для обучения
			args = training_args, # гиперпараметры для обучения (аргументы)
			train_dataset = dataset["train"], # тренировачный датасет
			eval_dataset = dataset["test"], # валидационный датасет
			data_collator = data_collator, # коллатор для подготовки обучения
			callbacks = [CustomCallback(writer)],
		)
	# top_k = 50 - выборка из 50 вероятных токенов, top_p - учитывание не менее 95% вероятности следующего слово 
	def generate_text(self, promt, max_length = 100, temperature = 0.5, top_k = 50, top_p = 0.9):
		"""Генерация текста по промпту"""
		try: 
			# Кодировка входного текста в последовательность ID токенов
			input_ids = self.tokenizer.encode(promt, return_tensors='pt').to(self.device)
			# Отключение вычисление градиентов для генерации
			with torch.no_grad():
				# Генерация текста с заданными параметрами
				output = self.model.generate(
					input_ids,
					max_length = max_length,
					temperature = temperature, # Контроль случайности
					top_k = top_k,
					top_p = top_p,
					repetition_penalty = 1.5, # Штраф за повторения
					do_sample = True, # Включение стохастической генерации
					pad_token_id = self.tokenizer.pad_token_id # ID токена заполнения
				)
			return self.tokenizer.decode(output[0], skip_special_tokens=True)
		except Exception as e:
			print(f"Generation error: {e}")
			return ""
	
	def save_model(self, output_dir):
		"""Сохранение модели"""
		# Создание директории, если она не существует
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# Сохранение модели и токенизатора
		self.model.save_pretrained(output_dir)
		self.tokenizer.save_pretrained(output_dir)

	def load_model(self, model_dir):
		"""Загрузка сохраненной модели"""
		try:
			self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
			self.model = GPT2LMHeadModel.from_pretrained(model_dir)
			print(f"Модель успешно загружена из {model_dir}")
			return True
		except Exception as e:
			print(f"Ошибка загрузки модели: {e}")
			return False
	


if __name__ == "__main__":
	writer = SummaryWriter()

	ai = MainAI()
	print(f"Используемое устройство: {ai.device}")
	print(f"Модель находится на {next(ai.model.parameters()).device}")
	dataset = ai.load_and_preparation_data("dataset_structured.txt", val_size = 0.1)
	print(f"Размер обучающего набора: {len(dataset['train'])}")
	print(f"Пример данных: {dataset['train'][0]}")
	trainer = ai.setup_trainer(dataset, "saved_model", writer)
	trainer.train()
	writer.close()
	print("Обучение завершено!")
	ai.save_model("saved_model")

