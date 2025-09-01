import os
import torch
import pickle
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


class SimpleTextGenerator:
    def __init__(
        self,
        model_name: str = "DeepESP/gpt2-spanish",
        output_dir: str = "models/fine_tuned_gpt2_spanish_New",
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length

        os.makedirs(self.output_dir, exist_ok=True)

        # Carga de tokenizer y modelo en español
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Historial de pérdidas
        self.history = {"train_loss": [], "eval_loss": [], "epoch": []}

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def prepare_dataset(self, texts: list[str]):
        ds = Dataset.from_dict({"text": texts})
        return ds.map(self._tokenize, batched=True)

    def fine_tune(
        self,
        train_texts: list[str],
        val_texts: list[str],
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 2e-5,
    ):
        train_ds = self.prepare_dataset(train_texts)
        eval_ds = self.prepare_dataset(val_texts)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir="logs",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True,
            report_to=["tensorboard"],  # Para desactivar integración con tensorboard escribir "none"
            logging_first_step=True,
            optim="adamw_torch_fused",
        )
        early_stop_callback = EarlyStoppingCallback(early_stopping_patience=0.7)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[early_stop_callback],
        )

        trainer.train()

        # Guardar modelo final y tokenizer
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Extraer historial
        for log in trainer.state.log_history:
            if "epoch" in log:
                epoch = log["epoch"]
                if "loss" in log:
                    self.history["epoch"].append(epoch)
                    self.history["train_loss"].append(log["loss"])
                if "eval_loss" in log:
                    self.history["eval_loss"].append(log["eval_loss"])

        # Guardar historial en pickle
        with open(f"{self.output_dir}/history.pkl", "wb") as f:
            pickle.dump(self.history, f)

    def generate(
        self,
        prompt: str,
        max_length: int = None,
        num_return_sequences: int = 1,
        temperature: float = 0.4,
    ) -> list[str]:
        max_len = max_length or self.max_length
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.8,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return [
            self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]


if __name__ == "__main__":
    # Ejemplo de uso
    texts = [
        "La inteligencia artificial está transformando la industria automotriz mediante sistemas de conducción autónoma que utilizan sensores láser y cámaras de alta resolución.",
        "Los algoritmos de machine learning pueden analizar patrones en grandes volúmenes de datos para predecir tendencias de mercado y comportamiento del consumidor.",
        "El procesamiento de lenguaje natural permite que las máquinas comprendan y generen texto humano de manera coherente y contextualmente relevante.",
        "Las redes neuronales convolucionales han revolucionado el reconocimiento de imágenes médicas, ayudando a detectar tumores con mayor precisión que los métodos tradicionales.",
        "Los sistemas de recomendación basados en inteligencia artificial personalizan la experiencia del usuario analizando su historial de navegación y preferencias.",
        "La robótica colaborativa está cambiando los procesos de manufactura al permitir que robots y humanos trabajen juntos de manera segura y eficiente.",   
        "Los chatbots conversacionales utilizan modelos de lenguaje avanzados para proporcionar atención al cliente las 24 horas del día con respuestas naturales.",
        "La visión por computadora permite que los vehículos autónomos identifiquen objetos, peatones y señales de tráfico en tiempo real para navegar de forma segura.",
        "Los algoritmos de deep learning pueden aprender patrones complejos en datos no estructurados como imágenes, audio y texto sin supervisión humana explícita.",
        "La automatización inteligente combina inteligencia artificial con robótica para optimizar procesos empresariales y reducir errores humanos significativamente."
    ]

    # División en entrenamiento y validación
    split = int(0.8 * len(texts))
    train_texts = texts[:split]
    val_texts = texts[split:]

    # Inicialización del generador
    generator = SimpleTextGenerator(
        model_name="DeepESP/gpt2-spanish",
        output_dir="models/fine_tuned_gpt2_spanish_New",
        max_length=256,
    )

    # Fine-tuning del modelo
    generator.fine_tune(
        train_texts=train_texts,
        val_texts=val_texts,
        epochs=20,
        batch_size=8,
        lr=2e-5,  # Valor estándar para Transformers
    )

    # Generación de texto
    sample = generator.generate(
        prompt="La inteligencia artificial ayuda a",
        max_length=20
    )
    print("Texto generado:", sample[0])


    sample = generator.generate("La inteligencia artificial sirve para", max_length=50)
    print("Texto generado:", sample[0])

