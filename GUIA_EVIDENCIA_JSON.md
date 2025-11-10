# 🎯 GUÍA: CONFIGURACIÓN DE MUESTRAS DEL DATASET

## ✅ NUEVO COMPORTAMIENTO

Cuando ejecutes `detector_zenodo_ultra_v3.py`, ahora te pregunta claramente:

```
⚙️  CONFIGURACIÓN DE MUESTRAS
======================================================================

¿Deseas limitar la cantidad de muestras por cada carpeta/clase?
   • SI: Te permite elegir cuántas muestras usar (ejemplo: 20, 50, 100)
   • NO: Usa TODAS las muestras disponibles en el dataset completo

¿Limitar muestras? (s/n):
```

## 📊 OPCIÓN 1: LIMITAR MUESTRAS (s)

Si respondes **"s"**, te pedirá cuántas muestras quieres por carpeta:

```
📊 Especifica el número de muestras por cada carpeta:
   Ejemplo: Si eliges 20, tomará:
   • 20 de Real/colombian
   • 20 de Real/chilean
   • 20 de StarGAN
   • 20 de CycleGAN
   • etc.

🎯 Muestras por carpeta: 20
```

### Resultado con 20 muestras:
- **20** audios de Real/colombian
- **20** audios de Real/chilean
- **20** audios de Real/peruvian
- **20** audios de Real/venezuelan
- **20** audios de Real/argentinian
- **20** audios de StarGAN
- **20** audios de CycleGAN
- **20** audios de Diffusion
- **20** audios de TTS
- **20** audios de TTS-StarGAN
- **20** audios de TTS-Diff

**Total aproximado**: 220 muestras (útil para pruebas rápidas)

### Uso recomendado:
- **20-50**: Pruebas muy rápidas (~5-10 minutos)
- **100-200**: Pruebas de concepto (~15-30 minutos)
- **500-1000**: Entrenamiento intermedio (~1-2 horas)
- **2000+**: Entrenamiento completo (~3-6 horas)

## 📦 OPCIÓN 2: DATASET COMPLETO (n)

Si respondes **"n"**, usará **TODAS** las muestras disponibles:

```
✅ Se usará el DATASET COMPLETO (todas las muestras disponibles)
```

### Resultado:
- **TODAS** las muestras de Real (ejemplo: 22,816)
- **TODAS** las muestras de StarGAN (ejemplo: 10,000)
- **TODAS** las muestras de CycleGAN (ejemplo: 10,000)
- **TODAS** las muestras de Diffusion (ejemplo: 8,000)
- **TODAS** las muestras de TTS (ejemplo: 15,000)
- **TODAS** las muestras de TTS-StarGAN (ejemplo: 7,500)
- **TODAS** las muestras de TTS-Diff (ejemplo: 7,500)

**Total**: ~80,816 muestras (dataset completo de Zenodo)

### Uso recomendado:
- Para **entrenamientos finales** de tu tesis
- Cuando necesitas **máxima precisión**
- Para **resultados publicables**
- Cuando tienes **tiempo suficiente** (6-12 horas)

## 🎓 RECOMENDACIONES PARA TU TESIS

### 1. Fase de Desarrollo
```bash
python detector_zenodo_ultra_v3.py
# Respuesta: s
# Muestras: 100
```
- Ideal para ajustar parámetros
- Detectar errores rápido
- Iterar sobre el código

### 2. Fase de Validación
```bash
python detector_zenodo_ultra_v3.py
# Respuesta: s
# Muestras: 1000
```
- Resultados preliminares confiables
- Tiempo de entrenamiento razonable
- Bueno para comparaciones

### 3. Fase Final (Para Tesis)
```bash
python detector_zenodo_ultra_v3.py
# Respuesta: n
```
- **Dataset completo** para resultados oficiales
- Métricas finales para incluir en tesis
- Máxima credibilidad académica

## 📈 COMPARACIÓN DE TIEMPOS

| Muestras | Tiempo Aprox. | Uso |
|----------|---------------|-----|
| 20 por carpeta (~220 total) | 5-10 min | Prueba rápida |
| 50 por carpeta (~550 total) | 10-20 min | Desarrollo |
| 100 por carpeta (~1,100 total) | 20-40 min | Validación |
| 500 por carpeta (~5,500 total) | 1-2 horas | Pre-final |
| 1000 por carpeta (~11,000 total) | 2-4 horas | Avanzado |
| **DATASET COMPLETO (~80,816)** | **6-12 horas** | **TESIS FINAL** |

## 💡 TIPS

### Para ahorrar tiempo:
1. **Primero prueba con 20-50 muestras** para verificar que todo funciona
2. Luego aumenta a 100-200 para validar
3. Finalmente, usa el dataset completo para resultados finales

### Para máxima precisión:
- Usa el **dataset completo** (sin límite)
- Ejecuta **múltiples entrenamientos** y compara
- Los archivos JSON guardarán todos los resultados

### Para debugging:
- Usa 20 muestras para encontrar errores rápido
- Los errores aparecerán igual con pocas o muchas muestras

## 📊 EJEMPLO DE EJECUCIÓN

```bash
python detector_zenodo_ultra_v3.py
```

**Salida esperada:**
```
======================================================================
🔬 DETECTOR ULTRA-AVANZADO V3 - ZENODO DATASET
======================================================================

🎯 Características:
   • 800+ características matemáticas avanzadas
   • Análisis Wavelet multi-nivel
   • Detección de artefactos GAN/TTS/VC
   • Análisis de fase y coherencia espectral
   • Microestructura temporal y prosódica
   • Ensemble de 10 algoritmos con Stacking
   • Optimizado para minimizar falsos negativos

📁 Ruta del dataset de Zenodo: C:\Users\johan\Downloads\dataset_zenodo

======================================================================
⚙️  CONFIGURACIÓN DE MUESTRAS
======================================================================

¿Deseas limitar la cantidad de muestras por cada carpeta/clase?
   • SI: Te permite elegir cuántas muestras usar (ejemplo: 20, 50, 100)
   • NO: Usa TODAS las muestras disponibles en el dataset completo

¿Limitar muestras? (s/n): s

📊 Especifica el número de muestras por cada carpeta:
   Ejemplo: Si eliges 20, tomará:
   • 20 de Real/colombian
   • 20 de Real/chilean
   • 20 de StarGAN
   • 20 de CycleGAN
   • etc.

🎯 Muestras por carpeta: 100

✅ Se usarán 100 muestras por cada carpeta/clase

🔍 Cargando Latin-American Voice Anti-Spoofing Dataset...
🎯 Límite por carpeta: 100 muestras
======================================================================
```

## 🎯 CONCLUSIÓN

- **"s" + número**: Control preciso de muestras (ideal para pruebas)
- **"n"**: Dataset completo (ideal para tesis final)
- **Flexibilidad total**: Tú decides según tus necesidades

**¡Ahora tienes control completo sobre cuántas muestras usar!** 🚀
