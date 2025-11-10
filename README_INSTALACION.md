# 📋 GUÍA DE INSTALACIÓN Y REQUISITOS
## Detector Zenodo Ultra V3 - Detección de Deepfakes de Voz

---

## 📦 REQUISITOS DEL SISTEMA

### Requisitos Mínimos
- **Sistema Operativo**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8 o superior (recomendado 3.9-3.11)
- **RAM**: Mínimo 8 GB (recomendado 16 GB para dataset completo)
- **Espacio en Disco**: 
  - 2 GB para librerías Python
  - 10-50 GB para el dataset (depende del tamaño)
  - 500 MB para modelos entrenados
- **Procesador**: Intel i5 / AMD Ryzen 5 o superior

### Requisitos Recomendados
- **RAM**: 16-32 GB
- **Procesador**: Intel i7 / AMD Ryzen 7 con 8+ núcleos
- **GPU**: No es obligatoria, pero acelera el entrenamiento

---

## 🐍 INSTALACIÓN DE PYTHON

### Windows
1. Descargar Python desde: https://www.python.org/downloads/
2. Durante la instalación, marcar "Add Python to PATH"
3. Verificar instalación:
   ```powershell
   python --version
   ```

### Linux/Ubuntu
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

### macOS
```bash
# Usando Homebrew
brew install python@3.11
python3 --version
```

---

## 📚 LIBRERÍAS REQUERIDAS

### 1. Instalación Automática (Recomendado)

Crea un archivo `requirements.txt` con el siguiente contenido:

```txt
# === CORE CIENTÍFICO ===
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# === MACHINE LEARNING ===
scikit-learn>=1.0.0
joblib>=1.1.0

# === AUDIO PROCESSING ===
librosa>=0.10.0
soundfile>=0.12.0
audioread>=3.0.0

# === WAVELETS Y ANÁLISIS AVANZADO ===
PyWavelets>=1.4.0
noisereduce>=2.0.0

# === VISUALIZACIÓN ===
matplotlib>=3.5.0
seaborn>=0.12.0

# === UTILIDADES ===
tqdm>=4.62.0
```

Instalar todas las dependencias:

```powershell
# Crear entorno virtual (recomendado)
python -m venv .venv

# Activar entorno virtual
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Instalación Manual (Paso a Paso)

Si prefieres instalar cada librería individualmente:

```powershell
# Core científico
pip install numpy scipy pandas

# Machine Learning
pip install scikit-learn joblib

# Audio processing
pip install librosa soundfile audioread

# Wavelets y análisis avanzado
pip install PyWavelets noisereduce

# Visualización
pip install matplotlib seaborn

# Utilidades
pip install tqdm
```

---

## 📊 DATASET REQUERIDO

### Latin-American Voice Anti-Spoofing Dataset (Zenodo)

#### Estructura del Dataset

El detector espera la siguiente estructura de carpetas:

```
dataset/
├── Real/                    # Voces humanas reales (bonafide)
│   ├── colombian/          # Audios .wav de hablantes colombianos
│   ├── chilean/            # Audios .wav de hablantes chilenos
│   ├── peruvian/           # Audios .wav de hablantes peruanos
│   ├── venezuelan/         # Audios .wav de hablantes venezolanos
│   └── argentinian/        # Audios .wav de hablantes argentinos
├── StarGAN/                # Deepfakes generados con StarGAN
├── CycleGAN/               # Deepfakes generados con CycleGAN
├── Diffusion/              # Deepfakes generados con Modelos de Difusión
├── TTS/                    # Voces sintéticas de Text-to-Speech
├── TTS-StarGAN/            # Híbrido TTS + StarGAN
└── TTS-Diff/               # Híbrido TTS + Diffusion
```

#### Dónde Obtener el Dataset

**Opción 1: Zenodo (Oficial)**
1. Visitar: https://zenodo.org/
2. Buscar: "Latin American Voice Spoofing" o "Voice Anti-Spoofing Spanish" https://zenodo.org/records/7370805 
3. Descargar el dataset completo (varios GB)
4. Extraer en una carpeta local

**Opción 2: Crear Dataset Propio**

Si no tienes acceso al dataset de Zenodo, puedes crear uno propio:

```
mi_dataset/
├── Real/              # Grabaciones de voces reales
│   └── *.wav         # Archivos de audio reales
└── Synthetic/         # Voces generadas por IA
    └── *.wav         # Archivos de audio sintéticos
```

**Requisitos de los archivos de audio:**
- Formato: `.wav` o `.mp3`
- Sample Rate: 16000 Hz (recomendado) o 22050 Hz
- Canales: Mono (1 canal)
- Duración: 2-10 segundos por archivo (ideal)

---

## 🚀 VERIFICACIÓN DE INSTALACIÓN

### Script de Verificación

Crea un archivo `verificar_instalacion.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para verificar que todas las dependencias están instaladas"""

import sys

def verificar_libreria(nombre, import_name=None):
    """Verifica si una librería está instalada"""
    if import_name is None:
        import_name = nombre
    
    try:
        __import__(import_name)
        print(f"✅ {nombre:20} - Instalado")
        return True
    except ImportError:
        print(f"❌ {nombre:20} - NO instalado")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DE DEPENDENCIAS")
    print("="*60 + "\n")
    
    librerias = [
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("Joblib", "joblib"),
        ("Librosa", "librosa"),
        ("SoundFile", "soundfile"),
        ("AudioRead", "audioread"),
        ("PyWavelets", "pywt"),
        ("NoiseReduce", "noisereduce"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("TQDM", "tqdm")
    ]
    
    resultados = []
    for nombre, import_name in librerias:
        resultado = verificar_libreria(nombre, import_name)
        resultados.append(resultado)
    
    print("\n" + "="*60)
    total = len(resultados)
    instalados = sum(resultados)
    
    if instalados == total:
        print(f"✅ TODAS LAS LIBRERÍAS INSTALADAS ({instalados}/{total})")
        print("="*60)
        print("\n🚀 ¡Sistema listo para entrenar el modelo!")
    else:
        faltantes = total - instalados
        print(f"⚠️  FALTAN {faltantes} LIBRERÍAS ({instalados}/{total})")
        print("="*60)
        print("\n📦 Instala las librerías faltantes con:")
        print("   pip install -r requirements.txt")
    
    # Verificar versión de Python
    print(f"\n🐍 Versión de Python: {sys.version}")
    if sys.version_info >= (3, 8):
        print("✅ Versión de Python compatible")
    else:
        print("⚠️  Se recomienda Python 3.8 o superior")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
```

Ejecutar verificación:

```powershell
python verificar_instalacion.py
```

---

## ▶️ CÓMO EJECUTAR EL MODELO

### 1. Preparar el Entorno

```powershell
# Activar entorno virtual (si lo usas)
.\.venv\Scripts\Activate.ps1

# Verificar instalación
python verificar_instalacion.py
```

### 2. Configurar Rutas

Asegúrate de tener:
- ✅ Dataset descargado y descomprimido
- ✅ Carpetas `Real/` y carpetas de ataques (`StarGAN/`, `TTS/`, etc.)

### 3. Ejecutar Entrenamiento

```powershell
python detector_zenodo_ultra_v3.py
```

**Durante la ejecución te preguntará:**

1. **Ruta del dataset:**
   ```
   📁 Ruta del dataset de Zenodo: C:\Users\tu_usuario\dataset_zenodo
   ```

2. **¿Limitar muestras?**
   - `s` → Entrenar con un subset (ej: 20, 50, 100 muestras por clase)
   - `n` → Usar TODO el dataset (recomendado para mejores resultados)

3. **¿Generar visualizaciones?**
   - `s` → Crea gráficas profesionales en `graficas_tesis/`
   - `n` → Solo entrena el modelo

### 4. Archivos Generados

Después del entrenamiento:

```
tesis/
├── detector_zenodo_ultra_v3.py          # Script principal
├── modelo_zenodo_ultra_v3.joblib        # Modelo entrenado (pickle)
├── resultados_entrenamiento/
│   ├── metricas_latest.json            # Últimas métricas
│   └── metricas_YYYYMMDD_HHMMSS.json   # Historial de métricas
└── graficas_tesis/
    ├── matriz_confusion.png            # Matriz de confusión
    ├── curva_roc.png                   # Curva ROC
    ├── precision_recall.png            # Curva Precision-Recall
    ├── comparativa_literatura.png      # Comparación con papers
    └── distribucion_confianza.png      # Histograma de confianza
```

---

## 🎯 USO DEL MODELO ENTRENADO

### Cargar Modelo y Predecir

```python
from detector_zenodo_ultra_v3 import DetectorZenodoUltraV3

# Crear instancia del detector
detector = DetectorZenodoUltraV3()

# Cargar modelo pre-entrenado
detector.cargar_modelo('modelo_zenodo_ultra_v3.joblib')

# Predecir en un audio nuevo
resultado = detector.predecir('audio_prueba.wav')

print(f"Es Deepfake: {resultado['is_deepfake']}")
print(f"Confianza: {resultado['confidence']:.2%}")
print(f"Probabilidad Real: {resultado['probability_real']:.2%}")
print(f"Probabilidad Deepfake: {resultado['probability_deepfake']:.2%}")
```

---

## 🔧 SOLUCIÓN DE PROBLEMAS COMUNES

### Error: "ModuleNotFoundError: No module named 'librosa'"

**Solución:**
```powershell
pip install librosa soundfile audioread
```

### Error: "Microsoft Visual C++ 14.0 is required" (Windows)

**Solución:**
1. Descargar e instalar "Microsoft C++ Build Tools"
2. Link: https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Reiniciar terminal y volver a ejecutar `pip install`

### Error: "MemoryError" durante entrenamiento

**Solución:**
1. Reducir el número de muestras usando el límite por clase
2. Cerrar otros programas que consuman RAM
3. Aumentar memoria virtual (swap) del sistema

### Error: "FileNotFoundError: Dataset not found"

**Solución:**
1. Verificar que la ruta del dataset sea correcta
2. Usar rutas absolutas: `C:\Users\...\dataset`
3. Verificar que existan carpetas `Real/` y las de ataques

### Audio no se carga: "Error loading audio file"

**Solución:**
```powershell
# Instalar dependencias de audio adicionales
pip install soundfile audioread

# En Linux también instalar:
sudo apt-get install libsndfile1 ffmpeg
```

---

## 📈 MÉTRICAS Y RESULTADOS

El modelo genera automáticamente:

### 1. Archivo JSON de Métricas

**Ubicación:** `resultados_entrenamiento/metricas_latest.json`

**Contiene:**
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Matriz de confusión (TP, TN, FP, FN)
- ✅ AUC-ROC, Cohen's Kappa, MCC
- ✅ False Negative Rate (crítico para seguridad)
- ✅ Comparación con literatura científica
- ✅ Configuración del modelo
- ✅ Timestamp y metadata

### 2. Visualizaciones Profesionales

**Ubicación:** `graficas_tesis/`

**Gráficas generadas:**
1. **Matriz de Confusión** - Errores de clasificación
2. **Curva ROC** - Trade-off sensibilidad/especificidad
3. **Precision-Recall** - Rendimiento en clases desbalanceadas
4. **Comparativa con Literatura** - Benchmark con papers
5. **Distribución de Confianza** - Histograma de predicciones

---

## 📚 RECURSOS ADICIONALES

### Documentación Oficial

- **Librosa:** https://librosa.org/doc/latest/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **NumPy:** https://numpy.org/doc/
- **Matplotlib:** https://matplotlib.org/

### Papers Relevantes

1. **Zhang et al. (2023)** - "Deep Learning for Voice Spoofing Detection"
2. **Wu et al. (2022)** - "Ensemble Methods for Deepfake Audio Detection"
3. **Kong et al. (2021)** - "MFCC-Based Features for Audio Deepfake Detection"

### Tutoriales

- Análisis de audio con Librosa: https://librosa.org/doc/latest/tutorial.html
- Machine Learning con scikit-learn: https://scikit-learn.org/stable/tutorial/

---

## 🆘 SOPORTE

### Problemas con el Código

1. Revisar la sección "Solución de Problemas Comunes"
2. Verificar versiones de librerías: `pip list`
3. Consultar documentación oficial de cada librería

### Problemas con el Dataset

1. Verificar estructura de carpetas
2. Confirmar que los archivos sean `.wav` válidos
3. Usar herramientas como `ffmpeg` para convertir formatos

### Errores de Memoria

1. Limitar muestras con la opción de límite por clase
2. Cerrar programas innecesarios
3. Considerar usar un servidor con más RAM

---

## ✅ CHECKLIST PRE-ENTRENAMIENTO

Antes de ejecutar `detector_zenodo_ultra_v3.py`, verifica:

- [ ] Python 3.8+ instalado
- [ ] Todas las librerías instaladas (`pip install -r requirements.txt`)
- [ ] Dataset descargado y descomprimido
- [ ] Estructura de carpetas correcta (Real/, StarGAN/, etc.)
- [ ] Al menos 8 GB de RAM disponible
- [ ] 10+ GB de espacio en disco libre
- [ ] Script `verificar_instalacion.py` ejecutado exitosamente

---

## 🎓 PARA TU TESIS

### Archivos Importantes para Incluir

1. **Metodología:**
   - `detector_zenodo_ultra_v3.py` (código fuente)
   - `README_DETECTOR_ZENODO_V3.md` (documentación técnica)
   - Este archivo (instalación y requisitos)

2. **Resultados:**
   - `resultados_entrenamiento/metricas_latest.json`
   - Todas las gráficas de `graficas_tesis/`
   - Tabla comparativa con literatura

3. **Evidencia:**
   - Logs de entrenamiento
   - Matriz de confusión
   - Curvas ROC y Precision-Recall
   - Comparación con estado del arte

---

## 📝 NOTAS FINALES

### Tiempo de Entrenamiento Estimado

- **Dataset pequeño (1,000 muestras):** 5-10 minutos
- **Dataset mediano (10,000 muestras):** 30-60 minutos
- **Dataset completo (80,000+ muestras):** 2-4 horas

### Recomendaciones

1. **Primera vez:** Entrenar con límite de 50-100 muestras para probar
2. **Entrenamiento final:** Usar dataset completo para mejores resultados
3. **Guardar modelos:** Cada entrenamiento guarda un nuevo modelo
4. **Backup:** Respaldar `modelo_zenodo_ultra_v3.joblib` y JSONs de métricas

### Próximos Pasos

1. ✅ Instalar dependencias
2. ✅ Verificar instalación
3. ✅ Descargar dataset
4. ✅ Entrenar modelo
5. ✅ Analizar resultados
6. ✅ Generar visualizaciones
7. ✅ Incluir en tesis

---

## 📞 INFORMACIÓN DE CONTACTO

Para soporte adicional sobre el detector, consulta:
- `README_DETECTOR_ZENODO_V3.md` - Documentación técnica completa
- `GUIA_EVIDENCIA_JSON.md` - Guía para interpretar resultados
- `GUIA_CONFIGURACION_MUESTRAS.md` - Optimización de muestras

---

**Versión:** 3.0  
**Última actualización:** Noviembre 2025  
**Compatibilidad:** Python 3.8 - 3.11  
**Licencia:** MIT (para uso académico)

---

