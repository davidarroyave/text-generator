import streamlit as st
import os
import pickle
import torch
from PIL import Image
from transformers import pipeline
import numpy as np
import datetime

# Configuración de la página
st.set_page_config(page_title="🤖 AI Transformer PLN", layout="wide")

# Sidebar con información de autores y descripción
with st.sidebar:
    st.title("💻 Autor")
    st.subheader("Desarrollado por:")
    st.markdown("Juan David Arroyave Ramirez") 
    st.markdown('https://davidarroyave.github.io', unsafe_allow_html=True)
    st.caption("Generative Model")
    st.caption("Creative Text Generator - NLP with Transformers")
    st.markdown("---")
    st.info(
        "Prototipo de aplicación para Procesamiento de Lenguaje Natural (PLN) usando GPT-2 (DeepESP/gpt2-spanish) tuneado a 20 epochs para generación de texto."
    )
    st.markdown("---")
    current_year = datetime.datetime.now().year
    st.markdown(f"""Modelo generador de texto creativo basado en DeepESP GPT-2 de Hugging Face. ©{current_year} Juan David Arroyave Ramirez. Licenciado bajo MIT: uso de software permitido según los términos de la licencia MIT. """)
    
    
# Título principal y descripción
st.title(" 🤖 Generador de Texto Creativo")
st.markdown(
    """
 Bienvenido al **generador de texto creativo con un modelo preentrenado transfomer de  Hugging Face  y desplegado con Streamlit**. 
- **Pestaña Generar Texto**: ingresa un prompt para generar salidas.
- **Pestaña Métricas**: visualiza curva de pérdida.
- **Pestaña Informe**: Introducción, marco teórico, solución y conclusiones.
"""
)

# Paths
MODEL_DIR = "models"
HISTORY_PATH = os.path.join(MODEL_DIR, "history.pkl")
OUTPUTS_DIR = "outputs"

# Cargar historial para métricas
history = {}
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "rb") as f:
        history = pickle.load(f)
else:
    st.warning("No se encontró el historial de entrenamiento (history.pkl).")

# Tabs
tab_generate, tab_metrics, tab_report = st.tabs(
    ["🎯 Generar Texto", "📊 Métricas del Modelo", "🧾 Informe"]
)

# ---------------------
# Tab 1: Generar Texto
# ---------------------
with tab_generate:
    st.subheader("🎯 Generación de Texto Creativo en Español")
    prompt = st.text_area(
        "Escribe tu prompt aquí:",
        "La inteligencia artificial es importante porque",
        height=120,
    )
    max_length = st.slider(
        "Máxima longitud de generación", min_value=1, max_value=50, value=10
    )
    num_return_sequences = st.slider(
        "Número de textos a generar", min_value=1, max_value=5, value=2
    )

    if st.button("Generar texto"):
        st.info("Generando… esto puede tardar unos segundos.")
        try:
            generator = pipeline(
                "text-generation",
                model=MODEL_DIR,
                tokenizer=MODEL_DIR,
                device=0 if torch.cuda.is_available() else -1,
            )
            
            outputs = generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=0.6,
                top_k=50,
                top_p=0.90,
                no_repeat_ngram_size=2,
        )
            for i, out in enumerate(outputs, start=1):
                st.markdown(f"**Texto generado #{i}:**")
                st.write(out["generated_text"])
                st.markdown("---")
        except Exception as e:
            st.error(f"Ocurrió un error al generar texto: {e}")

# --------------------------
# Tab 2: Métricas del Modelo
# --------------------------
with tab_metrics:
    st.subheader("📊 Métricas de Entrenamiento")

    loss_curve_path = os.path.join(OUTPUTS_DIR, "loss_curves.jpg")
    if os.path.exists(loss_curve_path):
        st.image(loss_curve_path, caption="Curvas de Pérdida", use_container_width=True)
    else:
        st.info("La curva de pérdida aún no está disponible. Ejecuta graficas.py.")

    if history and "train_loss" in history and len(history["train_loss"]) > 0:
        last_train_loss = history["train_loss"][-1]
        ppl = round(float(np.exp(last_train_loss)), 4)
        st.markdown(f"**Perplexity estimada (última época):** {ppl}")
    else:
        st.info(" La curva de pérdida obtenida baja consistentemente desde aproximadamente 4.8 hasta cerca de 2.5, lo que indica que el modelo está aprendiendo y ajustando sus parámetros correctamente en los datos de entrenamiento")

# ---------------------
# Tab 3: Informe
# ---------------------
with tab_report:
    st.subheader("🧾 Informe del Proyecto")

    st.markdown("## 1. Introducción")
    st.markdown(
        """
Los modelos de lenguaje basados en la arquitectura Transformer han revolucionado el campo del procesamiento de lenguaje natural (PLN) al permitir capturar dependencias a largo plazo mediante mecanismos de atención. GPT2-Spanish, desarrollado por DeepESP, es un modelo autoregresivo preentrenado específicamente en grandes cantidades de texto en español, cuya capacidad radica en predecir la siguiente palabra de una secuencia dada, generando así texto fluido y coherente. Gracias a su entrenamiento masivo, GPT2-Spanish cuenta con embeddings que codifican información semántica y sintáctica propia del idioma, lo que facilita tareas de completado, generación creativa y traducción dentro de su dominio lingüístico.

El proceso de transfer learning sobre GPT2-Spanish consiste en ajustar finamente sus pesos mediante fine-tuning sobre un corpus reducido y especializado, de manera que el modelo refuerce patrones específicos y reduzca comportamientos indeseados (como alucinaciones o cambios de idioma). Para ello, primero se tokeniza cada documento empleando la misma estrategia de segmentación de subpalabras utilizada en el preentrenamiento, se indexan las secuencias resultantes y se forman embeddings dinámicos que representan cada token en un espacio continuo de alta dimensión. Durante el entrenamiento, las capas de atención y feed-forward se refinan mediante retropropagación, optimizando la función de pérdida de entropía cruzada.

Este informe detalla el diseño de un prototipo en Streamlit que integra GPT2-Spanish afinado, despliega gráficas de pérdida versus época para monitorear el ajuste del modelo y permite a los usuarios generar texto creativo en español a partir de un prompt. La curva de pérdida obtenida baja consistentemente desde aproximadamente 4.8 hasta cerca de 2.5, lo que indica que el modelo está aprendiendo y ajustando sus parámetros correctamente en los datos de entrenamiento.
"""
    )

    st.markdown("## 2. Marco Teórico")
    st.markdown(
        """
**El Transformer** se basa en bloques de atención multi-cabeza (multi-head self-attention) y capas de feed-forward totalmente conectadas. En el mecanismo de atención, cada token produce queries, keys y values; la similitud entre queries y keys pondera los values, permitiendo al modelo enfocarse en partes relevantes de la secuencia. El uso de atención paralela elimina la necesidad de arquitecturas recurrentes, acelerando el entrenamiento y mejorando la captura de relaciones de largo alcance.

**La tokenización** de GPT2-Spanish se realiza mediante Byte-Pair Encoding (BPE), que divide el texto en subpalabras para balancear vocabulario y cobertura. Cada subpalabra recibe un índice entero y, al procesar una frase, se produce una secuencia de índices. La indexación y el embedding convierten estos índices en vectores densos de dimensión fija; estos vectores representan características semánticas y sintácticas aprendidas durante el preentrenamiento.

**El fine-tuning** adapta un modelo preentrenado a un dominio específico ajustando las capas finales y refinando las atenciones. Se emplea una función de pérdida de entropía cruzada que mide la discrepancia entre las predicciones y las etiquetas verdaderas. Para mitigar la sobreajuste, se utilizan técnicas como early stopping (detener entrenamiento cuando la métrica de validación deja de mejorar), weight decay y schedulers de tasa de aprendizaje (ReduceLROnPlateau). Estas estrategias garantizan que el modelo generalice fuera del conjunto de entrenamiento.
"""
    )

    st.markdown("## 3. Descripción del Problema")
    st.markdown(
        """
El objetivo del prototipo es proporcionar una herramienta de generación de texto en español que sea creativa, coherente y con un bajo nivel de alucinaciones. Se utilizo un modelo visto en clase, el GPT2-Spanish preentrenado, aunque especializado en español, puede incurrir en salidas erróneas cuando los prompts son amplios o cuando el modelo no ha visto suficiente variedad de ejemplos. Durante pruebas iniciales con 25 épocas de entrenamiento, la curva de validación descendía pero todavía mostraba repuntes y ejemplos alucinatorios en inglés. Esto evidenció la necesidad de un fine-tuning cuidadoso, con early stopping, reducción de learning rate y un corpus de entrenamiento más homogéneo en español.
"""
    )

    st.markdown("## 4. Planteamiento de la Solución")
    st.markdown(
        """
Para abordar el problema, se diseñó un pipeline de fine-tuning sobre GPT2-Spanish y un despliegue interactivo en Streamlit. Primero, se creó un entorno con PyTorch, Transformers y Datasets, y se hizo la particion en 80/20 para train y valid.  Luego, se entrenó el modelo durante hasta 50 épocas con un EarlyStoppingCallback (paciencia = 3), monitoreando eval_loss y reduciendo la tasa de aprendizaje mediante ReduceLROnPlateau. Almacenamos checkpoints por época y registramos historial de pérdida.

Se requirio de otro archivo .py para graficar las curvas de pérdida, tambien se desarrolló un dashboard en Streamlit con pestañas para generación de texto, visualización de métricas e informe técnico, integrando el modelo fine-tuneado y las gráficas generadas para interactuar de una manera amigable con la aplicación.

El objetivo es facilitar la portabilidad del mismo a traves de PyInstaller y permitir a usuarios sin conocimientos técnicos generar texto creativo y evaluar el desempeño del modelo a traves de una interfaz intuitiva y solo abriendo un ejecutable.
"""
    )

    st.markdown("## 5. Resultados")
    st.markdown(
        """
La gráfica Loss vs Epochs muestra un descenso continuo de la pérdida de validación desde 4.8 hasta cerca de 2.5, indicando que el modelo está aprendiendo patrones relevantes en el corpus de entrenamiento. La perplexity estimada en la última época es aproximadamente 11.95, lo que sugiere que el modelo tiene una buena capacidad para predecir la siguiente palabra en una secuencia dada. El loss de validación (Eval Loss) también baja, pero mucho más lento, estabilizándose alrededor de 5.0 después de la epoch 10. Este comportamiento es típico en modelos que generalizan razonablemente bien pero podrían beneficiarse de más datos variados o ajustes adicionales.

Un ejemplo de texto generado tras el fine-tuning es:

- Promt: La inteligencia artificial es importante porque

- Output: La mayoría de los sistemas informáticos son capaces de cambiar el mundo, pero la mayoría no lo son. Algunos han perdido el sentido de las matemáticas. Los ordenadores y las máquinas, sin embargo, han sido incapaces de comprender el funcionamiento de sus sistemas. A pesar de todo, los ordenadores se han convertido en ordenadores, y los teléfonos inteligentes han sobrevivido a la primera prueba de Turing. La idea básica de ordenadores es que la información de verdad está cambiando el universo, como un virus o una enfermedad. En las últimas décadas, las tecnologías han mejorado las condiciones de vida de otros sistemas, incluyendo los juegos informáticos. Se han hecho avances en la evolución y la tecnología han evolucionado a lo largo del tiempo. Las tecnologías modernas han avanzado en los primeros años de evolución, evolución e ingeniería. El desarrollo y el desarrollo han demostrado que las capacidades de software son mejores que los de ingeniería, ya que han ayudado a los tecnólogos a superar los retos y a aumentar las oportunidades. Un modelo de desarrollo avanzado es una teoría de inteligencia avanzada. Y ahora hay muchas más personas que intentan imitar las técnicas de pensamiento. Pero estos sistemas son diferentes. Muchos de ellos han aprendido a utilizar sistemas modernos en el pasado, desde la teoría informática hasta el diseño.”

Este texto refleja fluidez en español y coherencia temática con el prompt, aunque aún muestra ciertas repeticiones ligeras y expresiones redundantes, aspectos sujetos a refinamiento futuro.
"""
    )

    st.markdown("## 6. Conclusiones")
    st.markdown(
        """
El fine-tuning de GPT2-Spanish con early stopping y schedulers permitió reducir notablemente las alucinaciones y obtener muestras de texto coherente en español. No hay señal de overfitting severo, ya que la brecha entre train loss y eval loss no aumenta drásticamente. Sin embargo, el estancamiento de eval loss sugiere que el modelo ha llegado cerca de su límite de generalización para el dataset actual.

El descenso rápido del loss de entrenamiento frente al descenso lento de eval loss es normal, pero si la brecha sigue ampliándose, hay riesgo de sobreajuste.

El prototipo en Streamlit facilita la interacción de usuarios con el modelo, integrando generación de texto, visualización de métricas y documentación técnica en un solo dashboard. Esta arquitectura modular permite iterar sobre el corpus, ajustar hiperparámetros y desplegar nuevas versiones con mínima intervención.

Como trabajo futuro se recomienda enriquecer el dataset de fine-tuning con textos de dominio específico, aplicar técnicas de filtrado de datos y explorar arquitecturas de enmascaramiento o prompt engineering avanzado para mejorar aún más la coherencia y reducir repeticiones.
"""
    )

    st.markdown("## 7. Referencias")
    st.markdown(
        """
- A. Vaswani et al., “Attention Is All You Need,” in Advances in Neural Information Processing Systems, 2017, pp. 5998–6008.
- A. Radford et al., “Language Models are Unsupervised Multitask Learners,” OpenAI, 2019.
- DeepESP, “gpt2-spanish,” Hugging Face Models, 2025. Available: https://huggingface.co/DeepESP/gpt2-spanish
- T. Wolf et al., “Transformers: State-of-the-Art Natural Language Processing,” Proceedings of EMNLP, 2020.
"""
    )
