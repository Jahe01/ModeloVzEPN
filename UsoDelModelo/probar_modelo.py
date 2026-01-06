#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 SCRIPT PARA PROBAR EL MODELO ENTRENADO
Detecta si un audio es real o deepfake
Incluye grabación de audio desde el micrófono
Con historial de pruebas para comparativas
Genera visualizaciones y métricas detalladas
"""

import joblib
import sys
import os
import warnings
import tempfile
import numpy as np
from datetime import datetime
import json
import csv

# Suprimir warnings
warnings.filterwarnings('ignore')

# Para grabar audio
import sounddevice as sd
import soundfile as sf

# Para visualizaciones
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal, stats

# Importar el detector
from detector_zenodo_ultra_v3 import DetectorZenodoUltraV3

# Historial global de pruebas
historial_pruebas = []


def grabar_audio(duracion=5, sr=16000, descripcion=""):
    """
    Graba audio desde el micrófono
    
    Args:
        duracion: Duración en segundos
        sr: Sample rate (16000 Hz por defecto)
        descripcion: Descripción de las condiciones de grabación
    
    Returns:
        Ruta al archivo de audio grabado
    """
    print(f"\n🎙️ GRABACIÓN DE AUDIO")
    print("=" * 50)
    print(f"   Duración: {duracion} segundos")
    print(f"   Sample rate: {sr} Hz")
    if descripcion:
        print(f"   Condición: {descripcion}")
    print("-" * 50)
    
    input("Presiona ENTER para comenzar a grabar...")
    
    print("\n🔴 GRABANDO... ¡Habla ahora!")
    
    try:
        # Grabar audio
        audio = sd.rec(int(duracion * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()  # Esperar a que termine la grabación
        
        print("✅ Grabación completada!")
        
        # Guardar en archivo temporal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = os.path.join(tempfile.gettempdir(), f"grabacion_{timestamp}.wav")
        sf.write(temp_file, audio, sr)
        
        print(f"💾 Audio guardado en: {temp_file}")
        
        return temp_file
        
    except Exception as e:
        print(f"❌ Error al grabar: {e}")
        return None


def mostrar_historial():
    """Muestra el historial de todas las pruebas realizadas"""
    if not historial_pruebas:
        print("\n📭 No hay pruebas registradas aún.")
        return
    
    print("\n" + "=" * 110)
    print("📊 HISTORIAL DE PRUEBAS REALIZADAS")
    print("=" * 110)
    print(f"{'#':<4} {'Condición':<22} {'Duración':<10} {'Veredicto':<12} {'Real%':<9} {'Fake%':<9} {'Confianza':<11} {'Hora':<8}")
    print("-" * 110)
    
    for i, prueba in enumerate(historial_pruebas, 1):
        veredicto = "DEEPFAKE" if prueba['is_deepfake'] else "REAL"
        emoji = "🤖" if prueba['is_deepfake'] else "✅"
        duracion = f"{prueba.get('duracion', 0):.1f}s"
        condicion = prueba['condicion'][:20] + ".." if len(prueba['condicion']) > 22 else prueba['condicion']
        print(f"{i:<4} {condicion:<22} {duracion:<10} {emoji} {veredicto:<10} {prueba['prob_real']:<9.2f} {prueba['prob_fake']:<9.2f} {prueba['confianza']:<11.2f} {prueba['hora']:<8}")
    
    print("-" * 110)
    
    # Estadísticas
    total = len(historial_pruebas)
    reales = sum(1 for p in historial_pruebas if not p['is_deepfake'])
    deepfakes = sum(1 for p in historial_pruebas if p['is_deepfake'])
    confianza_promedio = sum(p['confianza'] for p in historial_pruebas) / total
    duracion_total = sum(p.get('duracion', 0) for p in historial_pruebas)
    duracion_promedio = duracion_total / total if total > 0 else 0
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   Total de pruebas: {total}")
    print(f"   Detectados como REAL: {reales} ({reales/total*100:.1f}%)")
    print(f"   Detectados como DEEPFAKE: {deepfakes} ({deepfakes/total*100:.1f}%)")
    print(f"   Confianza promedio: {confianza_promedio:.2f}%")
    print(f"   Duración total analizada: {duracion_total:.1f} segundos")
    print(f"   Duración promedio por audio: {duracion_promedio:.1f} segundos")
    print("=" * 110)


def exportar_historial():
    """Exporta el historial a CSV y JSON"""
    if not historial_pruebas:
        print("\n📭 No hay pruebas para exportar.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Exportar a CSV
    csv_file = f"resultados_pruebas_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['num', 'condicion', 'duracion_seg', 'veredicto', 'prob_real', 'prob_fake', 'confianza', 'hora', 'archivo'])
        writer.writeheader()
        for i, prueba in enumerate(historial_pruebas, 1):
            writer.writerow({
                'num': i,
                'condicion': prueba['condicion'],
                'duracion_seg': f"{prueba.get('duracion', 0):.2f}",
                'veredicto': 'DEEPFAKE' if prueba['is_deepfake'] else 'REAL',
                'prob_real': f"{prueba['prob_real']:.2f}",
                'prob_fake': f"{prueba['prob_fake']:.2f}",
                'confianza': f"{prueba['confianza']:.2f}",
                'hora': prueba['hora'],
                'archivo': prueba.get('archivo', 'N/A')
            })
    
    # Exportar a JSON
    json_file = f"resultados_pruebas_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'fecha_exportacion': datetime.now().isoformat(),
            'total_pruebas': len(historial_pruebas),
            'pruebas': historial_pruebas
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Historial exportado:")
    print(f"   📄 CSV: {csv_file}")
    print(f"   📄 JSON: {json_file}")


def generar_visualizaciones(audio_path, resultado, condicion="", output_dir="visualizaciones"):
    """
    Genera visualizaciones completas del audio analizado
    
    Args:
        audio_path: Ruta al archivo de audio
        resultado: Diccionario con resultados del análisis
        condicion: Descripción de la condición de grabación
        output_dir: Directorio para guardar las imágenes
    
    Returns:
        dict con métricas adicionales y rutas de imágenes
    """
    print("\n📊 Generando visualizaciones y métricas adicionales...")
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Timestamp para nombres únicos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calcular métricas adicionales
    metricas = calcular_metricas_audio(y, sr)
    
    # Configurar estilo de gráficos
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ============================================================
    # FIGURA 1: Panel completo de análisis (6 subplots)
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'🔬 ANÁLISIS DE AUDIO - {condicion}\nVeredicto: {"DEEPFAKE 🤖" if resultado["is_deepfake"] else "VOZ REAL ✅"} | Confianza: {resultado["confidence"]*100:.1f}%', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 1. Forma de onda (Waveform)
    ax1 = axes[0, 0]
    tiempo = np.linspace(0, len(y)/sr, len(y))
    ax1.plot(tiempo, y, color='#2196F3', linewidth=0.5, alpha=0.8)
    ax1.fill_between(tiempo, y, alpha=0.3, color='#2196F3')
    ax1.set_xlabel('Tiempo (s)', fontsize=10)
    ax1.set_ylabel('Amplitud', fontsize=10)
    ax1.set_title('📈 Forma de Onda (Waveform)', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, len(y)/sr])
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Espectrograma
    ax2 = axes[0, 1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
    ax2.set_title('🌈 Espectrograma', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frecuencia (Hz)', fontsize=10)
    ax2.set_xlabel('Tiempo (s)', fontsize=10)
    fig.colorbar(img, ax=ax2, format='%+2.0f dB', label='dB')
    
    # 3. Espectrograma Mel
    ax3 = axes[1, 0]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img2 = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax3, cmap='viridis')
    ax3.set_title('🎵 Espectrograma Mel', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frecuencia Mel', fontsize=10)
    ax3.set_xlabel('Tiempo (s)', fontsize=10)
    fig.colorbar(img2, ax=ax3, format='%+2.0f dB', label='dB')
    
    # 4. MFCCs
    ax4 = axes[1, 1]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img3 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax4, cmap='coolwarm')
    ax4.set_title('🔊 Coeficientes MFCC', fontsize=12, fontweight='bold')
    ax4.set_ylabel('MFCC #', fontsize=10)
    ax4.set_xlabel('Tiempo (s)', fontsize=10)
    fig.colorbar(img3, ax=ax4, label='Valor')
    
    # 5. Espectro de frecuencia promedio
    ax5 = axes[2, 0]
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    ax5.semilogy(freqs, fft, color='#4CAF50', linewidth=0.8)
    ax5.fill_between(freqs, fft, alpha=0.3, color='#4CAF50')
    ax5.set_xlabel('Frecuencia (Hz)', fontsize=10)
    ax5.set_ylabel('Magnitud (log)', fontsize=10)
    ax5.set_title('📊 Espectro de Frecuencia', fontsize=12, fontweight='bold')
    ax5.set_xlim([0, sr/2])
    ax5.axvline(x=metricas['frecuencia_fundamental'], color='red', linestyle='--', label=f'F0: {metricas["frecuencia_fundamental"]:.1f} Hz')
    ax5.legend()
    
    # 6. Características temporales (ZCR, Energía RMS)
    ax6 = axes[2, 1]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    frames = range(len(zcr))
    time_frames = librosa.frames_to_time(frames, sr=sr)
    
    ax6_twin = ax6.twinx()
    line1, = ax6.plot(time_frames, zcr, color='#FF5722', linewidth=1, label='Zero Crossing Rate')
    line2, = ax6_twin.plot(time_frames, rms, color='#9C27B0', linewidth=1, label='Energía RMS')
    ax6.set_xlabel('Tiempo (s)', fontsize=10)
    ax6.set_ylabel('ZCR', color='#FF5722', fontsize=10)
    ax6_twin.set_ylabel('RMS', color='#9C27B0', fontsize=10)
    ax6.set_title('⚡ Características Temporales', fontsize=12, fontweight='bold')
    ax6.legend([line1, line2], ['Zero Crossing Rate', 'Energía RMS'], loc='upper right')
    
    plt.tight_layout()
    
    # Guardar figura principal
    img_path_main = os.path.join(output_dir, f'analisis_completo_{timestamp}.png')
    plt.savefig(img_path_main, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Guardado: {img_path_main}")
    
    # ============================================================
    # FIGURA 2: Chromagram y características adicionales
    # ============================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f'🎼 ANÁLISIS ESPECTRAL DETALLADO - {condicion}', fontsize=14, fontweight='bold')
    
    # 1. Chromagram
    ax_chroma = axes2[0, 0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax_chroma, cmap='PuRd')
    ax_chroma.set_title('🎹 Chromagram', fontsize=11, fontweight='bold')
    
    # 2. Spectral Contrast
    ax_contrast = axes2[0, 1]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    librosa.display.specshow(contrast, x_axis='time', ax=ax_contrast, cmap='coolwarm')
    ax_contrast.set_title('🌊 Contraste Espectral', fontsize=11, fontweight='bold')
    ax_contrast.set_ylabel('Banda de frecuencia')
    
    # 3. Spectral Centroid, Bandwidth, Rolloff
    ax_spec = axes2[1, 0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    frames = range(len(centroid))
    t = librosa.frames_to_time(frames, sr=sr)
    
    ax_spec.plot(t, centroid, label='Centroide', color='blue', linewidth=1)
    ax_spec.plot(t, rolloff, label='Rolloff', color='red', linewidth=1)
    ax_spec.fill_between(t, centroid - bandwidth/2, centroid + bandwidth/2, alpha=0.3, label='Bandwidth')
    ax_spec.set_xlabel('Tiempo (s)')
    ax_spec.set_ylabel('Hz')
    ax_spec.set_title('📍 Características Espectrales', fontsize=11, fontweight='bold')
    ax_spec.legend(loc='upper right')
    
    # 4. Histograma de amplitudes
    ax_hist = axes2[1, 1]
    ax_hist.hist(y, bins=100, color='#3F51B5', alpha=0.7, edgecolor='white')
    ax_hist.axvline(x=np.mean(y), color='red', linestyle='--', label=f'Media: {np.mean(y):.4f}')
    ax_hist.axvline(x=np.std(y), color='green', linestyle='--', label=f'Std: {np.std(y):.4f}')
    ax_hist.set_xlabel('Amplitud')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.set_title('📊 Distribución de Amplitudes', fontsize=11, fontweight='bold')
    ax_hist.legend()
    
    plt.tight_layout()
    
    img_path_spectral = os.path.join(output_dir, f'analisis_espectral_{timestamp}.png')
    plt.savefig(img_path_spectral, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Guardado: {img_path_spectral}")
    
    # ============================================================
    # FIGURA 3: Resultado del modelo con gauge
    # ============================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gauge de probabilidad
    ax_gauge = axes3[0]
    prob_real = resultado['probability_real'] * 100
    prob_fake = resultado['probability_deepfake'] * 100
    
    colors = ['#4CAF50', '#F44336']
    sizes = [prob_real, prob_fake]
    labels = [f'Voz Real\n{prob_real:.1f}%', f'Deepfake\n{prob_fake:.1f}%']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax_gauge.pie(sizes, explode=explode, colors=colors, labels=labels,
                                            autopct='', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax_gauge.set_title(f'🎯 RESULTADO: {"DEEPFAKE 🤖" if resultado["is_deepfake"] else "VOZ REAL ✅"}', 
                       fontsize=14, fontweight='bold')
    
    # Barras de métricas clave
    ax_bars = axes3[1]
    metricas_mostrar = {
        'Confianza': resultado['confidence'] * 100,
        'F0 (Hz)': min(metricas['frecuencia_fundamental'], 100),  # Normalizado
        'SNR (dB)': min(max(metricas['snr_estimado'], 0), 100),  # Normalizado
        'Claridad': metricas['claridad_espectral'] * 100,
        'Dinamismo': metricas['rango_dinamico'] * 10
    }
    
    bars = ax_bars.barh(list(metricas_mostrar.keys()), list(metricas_mostrar.values()), 
                        color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4'])
    ax_bars.set_xlim([0, 100])
    ax_bars.set_xlabel('Valor (normalizado)', fontsize=10)
    ax_bars.set_title('📊 Métricas del Audio', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, metricas_mostrar.values()):
        ax_bars.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    img_path_resultado = os.path.join(output_dir, f'resultado_{timestamp}.png')
    plt.savefig(img_path_resultado, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Guardado: {img_path_resultado}")
    
    # Agregar rutas de imágenes a las métricas
    metricas['imagenes'] = {
        'analisis_completo': img_path_main,
        'analisis_espectral': img_path_spectral,
        'resultado': img_path_resultado
    }
    
    return metricas


def calcular_metricas_audio(y, sr):
    """
    Calcula métricas detalladas del audio
    
    Returns:
        dict con todas las métricas calculadas
    """
    metricas = {}
    
    # Duración
    metricas['duracion_segundos'] = len(y) / sr
    
    # Frecuencia fundamental (F0) usando autocorrelación
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        metricas['frecuencia_fundamental'] = np.mean(f0_valid) if len(f0_valid) > 0 else 0
        metricas['f0_std'] = np.std(f0_valid) if len(f0_valid) > 0 else 0
        metricas['f0_min'] = np.min(f0_valid) if len(f0_valid) > 0 else 0
        metricas['f0_max'] = np.max(f0_valid) if len(f0_valid) > 0 else 0
    except:
        metricas['frecuencia_fundamental'] = 0
        metricas['f0_std'] = 0
        metricas['f0_min'] = 0
        metricas['f0_max'] = 0
    
    # Energía RMS
    rms = librosa.feature.rms(y=y)[0]
    metricas['energia_rms_media'] = float(np.mean(rms))
    metricas['energia_rms_std'] = float(np.std(rms))
    metricas['energia_rms_max'] = float(np.max(rms))
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    metricas['zcr_media'] = float(np.mean(zcr))
    metricas['zcr_std'] = float(np.std(zcr))
    
    # Centroide espectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    metricas['centroide_espectral_media'] = float(np.mean(centroid))
    metricas['centroide_espectral_std'] = float(np.std(centroid))
    
    # Bandwidth espectral
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    metricas['bandwidth_media'] = float(np.mean(bandwidth))
    
    # Rolloff espectral
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    metricas['rolloff_media'] = float(np.mean(rolloff))
    
    # Flatness espectral
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    metricas['flatness_media'] = float(np.mean(flatness))
    
    # Contraste espectral
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    metricas['contraste_medio'] = float(np.mean(contrast))
    
    # SNR estimado
    signal_power = np.mean(y ** 2)
    noise_floor = np.percentile(np.abs(y), 10) ** 2
    metricas['snr_estimado'] = float(10 * np.log10(signal_power / (noise_floor + 1e-10)))
    
    # Rango dinámico
    metricas['rango_dinamico'] = float(np.max(np.abs(y)) - np.min(np.abs(y)))
    
    # Claridad espectral (entropía)
    S = np.abs(librosa.stft(y))
    S_norm = S / (np.sum(S) + 1e-10)
    entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10))
    metricas['claridad_espectral'] = float(1 / (1 + entropy/1000))  # Normalizado
    
    # Tempo estimado
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        metricas['tempo_estimado'] = float(tempo)
    except:
        metricas['tempo_estimado'] = 0
    
    # Estadísticas de la señal
    metricas['amplitud_media'] = float(np.mean(np.abs(y)))
    metricas['amplitud_max'] = float(np.max(np.abs(y)))
    metricas['amplitud_std'] = float(np.std(y))
    metricas['skewness'] = float(stats.skew(y))
    metricas['kurtosis'] = float(stats.kurtosis(y))
    
    # MFCCs promedio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        metricas[f'mfcc_{i+1}_media'] = float(np.mean(mfccs[i]))
    
    return metricas


def mostrar_metricas_detalladas(metricas):
    """Muestra las métricas de forma organizada"""
    print("\n" + "=" * 70)
    print("📊 MÉTRICAS DETALLADAS DEL AUDIO")
    print("=" * 70)
    
    print("\n🎵 CARACTERÍSTICAS TEMPORALES:")
    print(f"   • Duración: {metricas['duracion_segundos']:.2f} segundos")
    print(f"   • Energía RMS media: {metricas['energia_rms_media']:.6f}")
    print(f"   • Energía RMS máx: {metricas['energia_rms_max']:.6f}")
    print(f"   • Zero Crossing Rate: {metricas['zcr_media']:.4f}")
    
    print("\n🔊 CARACTERÍSTICAS DE VOZ:")
    print(f"   • Frecuencia Fundamental (F0): {metricas['frecuencia_fundamental']:.2f} Hz")
    print(f"   • F0 Desviación estándar: {metricas['f0_std']:.2f} Hz")
    print(f"   • F0 Rango: {metricas['f0_min']:.2f} - {metricas['f0_max']:.2f} Hz")
    
    print("\n📈 CARACTERÍSTICAS ESPECTRALES:")
    print(f"   • Centroide espectral: {metricas['centroide_espectral_media']:.2f} Hz")
    print(f"   • Bandwidth: {metricas['bandwidth_media']:.2f} Hz")
    print(f"   • Rolloff: {metricas['rolloff_media']:.2f} Hz")
    print(f"   • Flatness: {metricas['flatness_media']:.6f}")
    print(f"   • Contraste: {metricas['contraste_medio']:.4f}")
    
    print("\n📊 CALIDAD DE SEÑAL:")
    print(f"   • SNR estimado: {metricas['snr_estimado']:.2f} dB")
    print(f"   • Rango dinámico: {metricas['rango_dinamico']:.4f}")
    print(f"   • Claridad espectral: {metricas['claridad_espectral']:.4f}")
    
    print("\n📉 ESTADÍSTICAS:")
    print(f"   • Amplitud media: {metricas['amplitud_media']:.6f}")
    print(f"   • Amplitud máxima: {metricas['amplitud_max']:.6f}")
    print(f"   • Desviación estándar: {metricas['amplitud_std']:.6f}")
    print(f"   • Skewness: {metricas['skewness']:.4f}")
    print(f"   • Kurtosis: {metricas['kurtosis']:.4f}")
    
    print("\n🎼 MFCCs (primeros 5):")
    for i in range(5):
        print(f"   • MFCC {i+1}: {metricas[f'mfcc_{i+1}_media']:.4f}")
    
    print("=" * 70)


def cargar_modelo(modelo_path="modelo_zenodo_ultra_v3.joblib"):
    """Carga el modelo entrenado desde un archivo .joblib"""
    print(f"\n📂 Cargando modelo desde: {modelo_path}")
    
    if not os.path.exists(modelo_path):
        print(f"❌ Error: No se encontró el archivo {modelo_path}")
        return None
    
    try:
        modelo_data = joblib.load(modelo_path)
        
        # Crear instancia del detector
        detector = DetectorZenodoUltraV3()
        
        # Cargar componentes del modelo
        detector.model = modelo_data['model']
        detector.scaler = modelo_data['scaler']
        detector.feature_selector = modelo_data['feature_selector']
        detector.pca = modelo_data['pca']
        
        print("✅ Modelo cargado exitosamente!")
        print(f"   📊 Versión: {modelo_data.get('version', 'N/A')}")
        print(f"   📅 Fecha: {modelo_data.get('fecha_entrenamiento', 'N/A')}")
        
        return detector
        
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None


def analizar_audio(detector, audio_path, condicion="Sin especificar"):
    """Analiza un archivo de audio y muestra los resultados"""
    print(f"\n🎤 Analizando: {audio_path}")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: No se encontró el archivo {audio_path}")
        return
    
    # Calcular duración del audio
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duracion = len(y) / sr
    except:
        duracion = 0
    
    # Realizar predicción
    resultado = detector.predecir(audio_path)
    
    if 'error' in resultado:
        print(f"❌ Error: {resultado['error']}")
        return
    
    # Mostrar resultados
    print("\n📊 RESULTADOS DEL ANÁLISIS:")
    print("-" * 40)
    print(f"⏱️ Duración del audio: {duracion:.2f} segundos")
    
    if resultado['is_deepfake']:
        print("🤖 VEREDICTO: DEEPFAKE / VOZ SINTÉTICA")
    else:
        print("✅ VEREDICTO: VOZ REAL / HUMANA")
    
    print(f"\n📈 Probabilidades:")
    print(f"   • Voz Real:     {resultado['probability_real']*100:.2f}%")
    print(f"   • Voz Sintética: {resultado['probability_deepfake']*100:.2f}%")
    print(f"\n🎯 Confianza: {resultado['confidence']*100:.2f}%")
    
    # Interpretación de confianza
    conf = resultado['confidence']
    if conf >= 0.95:
        nivel_confianza = "Muy alta"
        print("   → Muy alta confianza en la predicción")
    elif conf >= 0.85:
        nivel_confianza = "Alta"
        print("   → Alta confianza en la predicción")
    elif conf >= 0.70:
        nivel_confianza = "Moderada"
        print("   → Confianza moderada")
    else:
        nivel_confianza = "Baja"
        print("   → ⚠️ Baja confianza - considerar análisis adicional")
    
    print("=" * 60)
    
    # Guardar en historial
    historial_pruebas.append({
        'condicion': condicion,
        'duracion': duracion,
        'is_deepfake': resultado['is_deepfake'],
        'prob_real': resultado['probability_real'] * 100,
        'prob_fake': resultado['probability_deepfake'] * 100,
        'confianza': resultado['confidence'] * 100,
        'nivel_confianza': nivel_confianza,
        'hora': datetime.now().strftime("%H:%M:%S"),
        'fecha': datetime.now().strftime("%Y-%m-%d"),
        'archivo': audio_path
    })
    
    return resultado


def analizar_audio_completo(detector, audio_path, condicion="Sin especificar", generar_viz=True):
    """
    Análisis completo con visualizaciones y métricas detalladas
    """
    # Primero hacer el análisis básico
    resultado = analizar_audio(detector, audio_path, condicion)
    
    if resultado is None or 'error' in resultado:
        return resultado
    
    # Generar visualizaciones y métricas
    if generar_viz:
        metricas = generar_visualizaciones(audio_path, resultado, condicion)
        mostrar_metricas_detalladas(metricas)
        
        # Actualizar el último registro del historial con las métricas
        if historial_pruebas:
            historial_pruebas[-1]['metricas'] = {k: v for k, v in metricas.items() if k != 'imagenes'}
            historial_pruebas[-1]['imagenes'] = metricas.get('imagenes', {})
    
    return resultado


def main():
    print("\n" + "=" * 60)
    print("🔬 DETECTOR DE DEEPFAKES - MODELO ULTRA V3")
    print("   Con visualizaciones y métricas detalladas")
    print("=" * 60)
    
    # Cargar modelo
    detector = cargar_modelo("modelo_zenodo_ultra_v3.joblib")
    
    if detector is None:
        return
    
    # Modo interactivo
    print("\n" + "=" * 60)
    print("📁 MODO DE PRUEBA CON HISTORIAL")
    print("=" * 60)
    print("\nOpciones:")
    print("  1. 🎙️ Grabar audio (con visualizaciones)")
    print("  2. 📁 Analizar archivo (con visualizaciones)")
    print("  3. 📂 Analizar carpeta completa")
    print("  4. 📊 Ver historial de pruebas")
    print("  5. 💾 Exportar historial (CSV/JSON)")
    print("  6. 🧪 Modo pruebas múltiples (comparativa)")
    print("  7. 🚪 Salir")
    
    while True:
        print("\n" + "-" * 40)
        opcion = input("Selecciona una opción (1-7): ").strip()
        
        if opcion == "1":
            # Grabar audio con descripción de condición
            print("\n📋 CONDICIONES DE GRABACIÓN SUGERIDAS:")
            print("   1. Silencio total")
            print("   2. Ruido de fondo leve")
            print("   3. Ruido de fondo moderado")
            print("   4. Con música de fondo")
            print("   5. Micrófono de laptop")
            print("   6. Micrófono externo/auriculares")
            print("   7. Otra (especificar)")
            
            cond = input("\nSelecciona condición (1-7) o escribe tu propia descripción: ").strip()
            condiciones_map = {
                '1': 'Silencio total',
                '2': 'Ruido de fondo leve',
                '3': 'Ruido de fondo moderado',
                '4': 'Con música de fondo',
                '5': 'Micrófono de laptop',
                '6': 'Micrófono externo',
            }
            condicion = condiciones_map.get(cond, cond if cond != '7' else input("Describe la condición: ").strip())
            
            try:
                duracion = input("\n⏱️ Duración en segundos (default 5): ").strip()
                duracion = int(duracion) if duracion else 5
            except:
                duracion = 5
            
            audio_path = grabar_audio(duracion=duracion, descripcion=condicion)
            if audio_path:
                # Preguntar si quiere visualizaciones
                gen_viz = input("\n📊 ¿Generar visualizaciones y métricas? (s/n): ").strip().lower() == 's'
                
                if gen_viz:
                    analizar_audio_completo(detector, audio_path, condicion=condicion, generar_viz=True)
                else:
                    analizar_audio(detector, audio_path, condicion=condicion)
                
                # Preguntar si quiere guardar el audio
                guardar = input("\n💾 ¿Guardar la grabación? (s/n): ").strip().lower()
                if guardar == 's':
                    nombre = input("📝 Nombre del archivo (sin extensión): ").strip()
                    if nombre:
                        destino = f"{nombre}.wav"
                        import shutil
                        shutil.copy(audio_path, destino)
                        print(f"✅ Audio guardado como: {destino}")
        
        elif opcion == "2":
            audio_path = input("\n🎵 Ruta del archivo de audio (.wav, .mp3, .flac): ").strip()
            audio_path = audio_path.strip('"').strip("'")  # Quitar comillas
            condicion = input("📋 Descripción/condición del audio (opcional): ").strip() or "Archivo externo"
            
            # Preguntar si quiere visualizaciones
            gen_viz = input("📊 ¿Generar visualizaciones y métricas? (s/n): ").strip().lower() == 's'
            
            if gen_viz:
                analizar_audio_completo(detector, audio_path, condicion=condicion, generar_viz=True)
            else:
                analizar_audio(detector, audio_path, condicion=condicion)
            
        elif opcion == "3":
            carpeta = input("\n📂 Ruta de la carpeta con audios: ").strip()
            carpeta = carpeta.strip('"').strip("'")
            
            if not os.path.isdir(carpeta):
                print(f"❌ No es una carpeta válida: {carpeta}")
                continue
            
            # Buscar archivos de audio
            extensiones = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            archivos = []
            for archivo in os.listdir(carpeta):
                if any(archivo.lower().endswith(ext) for ext in extensiones):
                    archivos.append(os.path.join(carpeta, archivo))
            
            if not archivos:
                print("❌ No se encontraron archivos de audio en la carpeta")
                continue
            
            print(f"\n📊 Encontrados {len(archivos)} archivos de audio")
            condicion = input("📋 Descripción común para estos audios: ").strip() or "Carpeta"
            gen_viz = input("📊 ¿Generar visualizaciones para cada audio? (s/n): ").strip().lower() == 's'
            
            # Analizar cada archivo
            resultados = {'real': 0, 'deepfake': 0}
            for audio_path in archivos:
                cond_archivo = f"{condicion}: {os.path.basename(audio_path)}"
                if gen_viz:
                    resultado = analizar_audio_completo(detector, audio_path, condicion=cond_archivo, generar_viz=True)
                else:
                    resultado = analizar_audio(detector, audio_path, condicion=cond_archivo)
                
                if resultado and 'error' not in resultado:
                    if resultado['is_deepfake']:
                        resultados['deepfake'] += 1
                    else:
                        resultados['real'] += 1
            
            # Resumen
            print("\n" + "=" * 60)
            print("📊 RESUMEN DEL ANÁLISIS DE CARPETA")
            print("=" * 60)
            print(f"   Total analizados: {len(archivos)}")
            print(f"   ✅ Voces reales:    {resultados['real']}")
            print(f"   🤖 Deepfakes:       {resultados['deepfake']}")
        
        elif opcion == "4":
            mostrar_historial()
        
        elif opcion == "5":
            exportar_historial()
        
        elif opcion == "6":
            # Modo pruebas múltiples
            print("\n" + "=" * 60)
            print("🧪 MODO PRUEBAS MÚLTIPLES - COMPARATIVA")
            print("=" * 60)
            print("\nEste modo te permite hacer varias grabaciones seguidas")
            print("con diferentes condiciones para comparar resultados.\n")
            
            condiciones_sugeridas = [
                "Silencio total",
                "Ruido de fondo leve", 
                "Ruido de fondo moderado",
                "Con música de fondo",
                "Hablando cerca del mic",
                "Hablando lejos del mic"
            ]
            
            print("Condiciones sugeridas para probar:")
            for i, c in enumerate(condiciones_sugeridas, 1):
                print(f"   {i}. {c}")
            
            try:
                num_pruebas = int(input("\n¿Cuántas pruebas quieres hacer? ").strip())
            except:
                num_pruebas = 3
            
            try:
                duracion = int(input("⏱️ Duración por prueba en segundos (default 5): ").strip() or "5")
            except:
                duracion = 5
            
            for i in range(num_pruebas):
                print(f"\n{'='*50}")
                print(f"📍 PRUEBA {i+1} de {num_pruebas}")
                print(f"{'='*50}")
                condicion = input(f"Describe la condición de esta prueba: ").strip()
                
                audio_path = grabar_audio(duracion=duracion, descripcion=condicion)
                if audio_path:
                    analizar_audio(detector, audio_path, condicion=condicion)
                
                if i < num_pruebas - 1:
                    input("\nPresiona ENTER para continuar con la siguiente prueba...")
            
            print("\n" + "=" * 60)
            print("✅ PRUEBAS COMPLETADAS")
            mostrar_historial()
            
            exportar = input("\n¿Exportar resultados? (s/n): ").strip().lower()
            if exportar == 's':
                exportar_historial()
            
        elif opcion == "7":
            if historial_pruebas:
                guardar = input("\n💾 ¿Exportar historial antes de salir? (s/n): ").strip().lower()
                if guardar == 's':
                    exportar_historial()
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida. Usa 1-7")


if __name__ == "__main__":
    # Si se pasa un argumento, analizar ese archivo directamente
    if len(sys.argv) > 1:
        detector = cargar_modelo("modelo_zenodo_ultra_v3.joblib")
        if detector:
            for audio in sys.argv[1:]:
                analizar_audio(detector, audio)
    else:
        main()
