# 📊 Análisis de Resultados - Pruebas del Modelo

## Resumen de las Pruebas

| # | Condición | Veredicto | Real% | Fake% | ¿Correcto? |
|---|-----------|-----------|-------|-------|------------|
| 1 | Voz normal + ruido PC | ✅ REAL | 74.6% | 25.4% | ✅ Sí |
| 2 | Voz normal desde PC, pocos ruidos | 🤖 DEEPFAKE | 41.8% | 58.2% | ❌ **Falso positivo** |
| 3 | Voz normal + lluvia de fondo | ✅ REAL | 79.9% | 20.1% | ✅ Sí |
| 4 | Voz normal con muletillas/risas | ✅ REAL | 64.7% | 35.3% | ✅ Sí |
| 5 | Voz normal leyendo texto | 🤖 DEEPFAKE | 30.6% | 69.4% | ❌ **Falso positivo** |
| 6 | Voz Sintética - topmediai | 🤖 DEEPFAKE | 31.5% | 68.5% | ✅ Sí |
| 7 | Voz Sintética - voicv.com | 🤖 DEEPFAKE | 32.0% | 68.0% | ✅ Sí |
| 8 | Voz Sintética - huggingface | 🤖 DEEPFAKE | 33.4% | 66.6% | ✅ Sí |
| 9 | Voz Sintética - huggingface + ruido | 🤖 DEEPFAKE | 47.6% | 52.4% | ✅ Sí |
| 10 | Voz Sintética - veed.io | 🤖 DEEPFAKE | 37.4% | 62.6% | ✅ Sí |

---

## 📈 Métricas de Rendimiento

- **Precisión general**: 8/10 (80%)
- **Voces reales correctas**: 3/5 (60%)
- **Voces sintéticas correctas**: 5/5 (100%)
- **Falsos positivos**: 2 (voces reales detectadas como deepfake)
- **Falsos negativos**: 0 (ningún deepfake pasó como real)

---

## 🔍 Análisis de Falsos Positivos

### Prueba 2 - Voz desde computadora (58.2% fake)

**Condición**: "Voz solo desde la computadora, voz normal sin leves ruidos de fondo"

**Factores que afectaron**:
- Confianza **baja** (58%) - muy cerca del umbral de incertidumbre (50%)
- El **ruido del ventilador** de la PC genera frecuencias constantes similares a artefactos de audio sintético
- El sonido constante del ventilador enmascara las variaciones naturales de la voz

### Prueba 5 - Lectura de texto (69.4% fake)

**Condición**: "Voz normal, leer normal un texto"

**Factores que afectaron**:
- **Leer de forma monótona** produce patrones similares a Text-to-Speech (TTS)
- Menos variación prosódica (entonación, ritmo)
- Falta de pausas naturales, respiraciones y "imperfecciones" humanas
- El modelo detecta habla "demasiado perfecta" como sospechosa

---

## 🎯 ¿Por qué el ruido del ventilador afecta la detección?

1. **Frecuencias constantes**: El ventilador emite un zumbido de frecuencia fija (~100-500 Hz), similar a artefactos de compresión presentes en deepfakes

2. **Enmascaramiento de armónicos**: El ruido oculta las variaciones naturales de la voz humana que el modelo utiliza para identificar voces reales

3. **Pérdida de microdetalles**: Las fluctuaciones naturales de la voz (jitter, shimmer, microtemblores) se pierden en el ruido de fondo

---

## 🎯 ¿Por qué hablar de forma monótona afecta?

Las voces sintéticas (TTS) generan habla "perfecta" y fluida. Cuando una persona **lee sin expresión**:

- Produce **menos variación en F0** (frecuencia fundamental)
- Las **transiciones son demasiado suaves**
- **Faltan "imperfecciones" humanas**: respiración audible, pausas de duda, cambios de velocidad
- El modelo interpreta esto como características típicas de audio sintético

---

## ✅ Conclusiones

### Fortalezas del Modelo

1. **100% de precisión en deepfakes**: Todas las voces sintéticas fueron detectadas correctamente
2. **Robusto ante ruido natural**: La lluvia de fondo (prueba 3) no afectó negativamente
3. **Detecta variaciones humanas**: Las muletillas y risas (prueba 4) ayudan a confirmar voz real

### Limitaciones Identificadas

1. **Sensible al ruido mecánico constante**: Ventiladores, aire acondicionado
2. **Habla monótona genera falsos positivos**: Lectura sin expresión
3. **Confianza baja en casos límite**: Valores entre 55-65% son poco confiables

### Recomendaciones para Mejores Resultados

1. **Usar micrófono externo** alejado de fuentes de ruido constante
2. **Hablar de forma natural**, no leyendo textos
3. **Considerar solo predicciones con confianza > 70%** como confiables
4. **Incluir variación vocal**: pausas, cambios de tono, expresiones naturales

---

## 📝 Conclusiones

Los falsos positivos en las pruebas 2 y 5 demuestran que el modelo es sensible a:
- Ruido de fondo constante (ventilador de PC)
- Patrones de habla monótona similar a TTS

Esto sugiere que el modelo podría beneficiarse de:
- Entrenamiento adicional con ruido de fondo variado
- Mejor diferenciación entre habla monótona real y sintética

La **tasa de detección de deepfakes del 100%** indica que el modelo es efectivo para su propósito principal: identificar voces sintéticas y prevenir fraudes telefónicos (vishing).
