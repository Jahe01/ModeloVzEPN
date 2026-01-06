# 🎯 Uso del Modelo - Detector de Deepfakes de Voz

Guía rápida para probar el modelo de detección de voces sintéticas.

## 📦 Requisitos

```bash
pip install numpy==2.1.0 scikit-learn==1.7.2 librosa sounddevice soundfile matplotlib scipy
```

## 🚀 Uso

```bash
python probar_modelo.py
```

## 📋 Opciones del Menú

| Opción | Descripción |
|--------|-------------|
| 1 | Grabar audio desde micrófono |
| 2 | Analizar archivo de audio |
| 3 | Analizar carpeta completa |
| 4 | Ver historial de pruebas |
| 5 | Exportar historial (CSV/JSON) |
| 6 | Modo pruebas múltiples |
| 7 | Salir |

## 📊 Resultados

- **Historial**: Exportable a CSV o JSON
- **Métricas**: Duración, confianza, probabilidades

## ⚠️ Errores Comunes

```bash
# Si hay error de NumPy/scikit-learn:
pip install numpy==2.1.0 scikit-learn==1.7.2
```

