import numpy as np
import pandas as pd
import ruptures as rpt
import plotly.graph_objects as go
from ruptures.base import BaseCost
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import ks_2samp,wasserstein_distance

class AnomalyDetection:

    #This class accepts...
    def __init__(self, df, windowSize, overlapWindow):
        self.df = df
        self.windowSize = windowSize
        self.overlapWindow = overlapWindow
    
    def merge_close_breakpoints(self, breakpoints, threshold=10):
        """Unisce i breakpoints vicini in un unico punto rappresentativo."""
        if not breakpoints:
            return []
        
        merged_breakpoints = []
        current_group = [breakpoints[0]]

        for bp in breakpoints[1:]:
            if bp - current_group[-1] <= threshold:
                current_group.append(bp)
            else:
                merged_breakpoints.append(int(np.mean(current_group)))  # Usa la media o la mediana
                current_group = [bp]
        
        if current_group:
            merged_breakpoints.append(int(np.mean(current_group)))

        return merged_breakpoints
    
    def test(self, penalty, minSize):
        # Estrazione dei dati per l'analisi
        #signal = np.vstack([self.df[column].values]).T

        # Genera i tre segmenti
        segment1 = np.zeros(100)  # 100 valori a 0
        segment2 = np.full(100, 10)  # 100 valori a +10
        segment3 = np.full(50, -20)  # 100 valori a -20

        # Unisce i segmenti in un unico segnale
        signal = np.concatenate([segment1, segment2, segment3])
        signal = signal.reshape(-1,1)
        # Creazione del modello PELT con la funzione di costo personalizzata
        cost = CostKLStats(minSize=minSize).fit(signal)
        breakpoints = cost.detect_anomalies(penalty)

        for breakpoint in breakpoints:
            print(cost.get_point_explanation(breakpoint))

        # Crea il grafico con Plotly
        fig = go.Figure()

        # Traccia il segnale scalato
        fig.add_trace(go.Scatter(
            x=np.arange(len(signal)),
            y=signal.flatten(), 
            mode='lines', 
            name='Signal Scaled'
        ))

        for breakpoint in breakpoints:
            fig.add_shape(
                type="line",
                x0=breakpoint, x1=breakpoint,
                y0=-25, y1=15,
                line=dict(color="red", width=2, dash="dash")  # Linea rossa tratteggiata
            )

        # Personalizza il layout
        fig.update_layout(
            title=f"Change Points on test",
            xaxis_title="Day",
            yaxis_title="Time",
            #template="plotly_dark",
            showlegend=True
        )

        # Mostra il grafico
        fig.show()

        rpt.display(signal,breakpoints)

    def start(self, column, penalty, minSize):
        # Estrazione dei dati per l'analisi
        dates = self.df['date'].values
        signal = self.df[column].values
        signal = signal.reshape(-1,1)
        
        # Creazione del modello PELT con la funzione di costo personalizzata
        cost = CostKLStats(minSize=minSize).fit(signal)
        breakpoints = cost.detect_anomalies(penalty)

        for breakpoint in breakpoints:
            print(cost.get_point_explanation(breakpoint))

        # Crea il grafico con Plotly
        fig = go.Figure()

        # Traccia il segnale scalato
        fig.add_trace(go.Scatter(
            x=dates,
            y=signal.flatten(), 
            mode='lines', 
            name='Signal Scaled'
        ))

        for breakpoint in breakpoints:
            fig.add_shape(
                type="line",
                x0=dates[breakpoint-1], x1=dates[breakpoint-1],
                y0=np.min(signal)-1, y1=np.max(signal)+1,
                line=dict(color="red", width=2, dash="dash")  # Linea rossa tratteggiata
            )

        # Personalizza il layout
        fig.update_layout(
            title=f"Change Points on {column}",
            xaxis_title="Day",
            yaxis_title="Time",
            #template="plotly_dark",
            showlegend=True
        )

        # Mostra il grafico
        fig.show()

        rpt.display(signal,breakpoints)

    def startWindow(self, column, penalty, minSize, aggregatedDays=-1, percentChangePoints = 0.5):
        if aggregatedDays == -1:
            aggregatedDays = minSize
        # Estrazione dei dati per l'analisi
        signal = self.df[column].values
        signal = signal.reshape(-1,1)
        
        # Inizializzazione delle variabili
        start_idx = 0
        end_idx = self.windowSize
        breakpoints_dict = {}

        overlap = self.windowSize - self.overlapWindow

        while end_idx <= len(signal):
            # Estrazione della ground truth e del segnale da controllare
            groundTruth = signal[start_idx:end_idx]
            signalToCheck = signal[start_idx + overlap + 1 :]

            if len(signalToCheck) < minSize:
                break  # Oppure continua con un'altra gestione

            # Creazione del modello PELT con la funzione di costo personalizzata
            cost = CostKLStats(minSize=minSize).fit(groundTruth)
            model = rpt.Pelt(custom_cost=cost, min_size=minSize).fit(signalToCheck)

            # Predizione dei punti di cambiamento
            breakpoints = model.predict(pen=penalty)

            # Aggiunta dei punti di cambiamento al dizionario (aggiustati per l'indice corrente)
            adjusted_breakpoints = [bp + start_idx + overlap for bp in breakpoints]
            for bp in adjusted_breakpoints:
                breakpoints_dict[bp] = breakpoints_dict.get(bp, 0) + 1
            
            # Aggiornamento degli indici per la prossima iterazione
            start_idx = start_idx + overlap + 1
            end_idx = end_idx + overlap + 1
        
        # Unione dei breakpoint vicini entro la soglia specificata
        sorted_keys = sorted(breakpoints_dict.keys())
        merged_breakpoints = {}
        
        while sorted_keys:
            ref_bp = sorted_keys.pop(0)
            count = breakpoints_dict[ref_bp]
            to_merge = [ref_bp]
            
            while sorted_keys and sorted_keys[0] <= ref_bp + aggregatedDays:
                to_merge.append(sorted_keys.pop(0))
            
            merged_bp = round(sum(to_merge) / len(to_merge))  # Media arrotondata dei breakpoint
            merged_breakpoints[merged_bp] = sum(breakpoints_dict[bp] for bp in to_merge)
        
        final_breakpoints = {}
        for key, value in merged_breakpoints.items():
            if(key // overlap * percentChangePoints < value):
                final_breakpoints[key] = value
        
        # Crea il grafico con Plotly
        fig = go.Figure()

        # Traccia il segnale scalato
        fig.add_trace(go.Scatter(
            x=np.arange(len(signal)), 
            y=signal.flatten(), 
            mode='lines', 
            name='Signal Scaled'
        ))

        for breakpoint in final_breakpoints.keys():
            # Aggiungi una linea verticale a x = 50
            fig.add_shape(
                type="line",
                x0=breakpoint, x1=breakpoint,  # Linea fissa a x = 50
                y0=-2, y1=2,  # Estende la linea sull'asse Y (da 0 a 1)
                line=dict(color="red", width=2, dash="dash")  # Linea rossa tratteggiata
            )

        # Personalizza il layout
        fig.update_layout(
            title=f"Change Points on {column}",
            xaxis_title="Day",
            yaxis_title="Time",
            #template="plotly_dark",
            showlegend=True
        )

        # Mostra il grafico
        fig.show()

        # Visualizza il grafico
        #plt.figure(figsize=(10, 6))
        #rpt.display(signal_scaled, final_breakpoints.keys())
        #plt.xlabel("Day")
        #plt.ylabel("Time")
        #plt.show()
    
class CostKLStats(BaseCost):
    """Costo basato sulla KL-divergence tra media, deviazione standard e varianza."""
    
    def __init__(self, minSize):
        super().__init__()
        self.global_stats = None  # Statistiche del segnale intero
        self.min_size = minSize
        self.signal = None
        self.mean_signal = None
        self.std_signal = None
        self.var_signal = None
        self.epsilon = 1e-10
        self.anomaly_details = {}  # Dizionario per memorizzare spiegazioni
        self.change_points = []  # Per memorizzare i segmenti rilevati
    
    def fit(self, signal):
        """Memorizza il segnale e calcola le statistiche globali."""
        self.signal = signal  # Convertire in array 1D
        self.mean_signal = np.mean(signal)
        self.std_signal = np.std(signal)
        self.var_signal = np.var(signal)
        self.max_signal = np.max(self.signal)
        self.min_signal = np.min(self.signal)
        return self

    def error(self, start, end):
        
        segment = self.signal[start:end]

        mean_segment = np.mean(segment)
        std_segment = np.std(segment)
        var_segment = np.var(segment)
        max_segment = np.max(segment)
        min_segment = np.min(segment)
        mean_diff = abs(mean_segment - self.mean_signal)
        std_diff = abs(std_segment - self.std_signal)
        var_diff = abs(var_segment - self.var_signal)
        max_diff = abs(max_segment - self.max_signal)
        min_diff = abs(min_segment - self.min_signal)
        

        # KS Test
        ks_stat, _ = ks_2samp(segment, self.signal)

        # Earth Mover's Distance (EMD)
        emd = wasserstein_distance(np.ravel(segment), np.ravel(self.signal))

        # Punteggio combinato
        cost = (ks_stat[0] + emd + mean_diff + std_diff + var_diff + max_diff + min_diff)*-1
        
        # Salva la spiegazione nel dizionario
        self.anomaly_details[(start, end)] = {
            "cost": cost,
            "ks_value": ks_stat[0],
            "emd_value": emd,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "var_diff": var_diff,
            "max_diff": max_diff,
            "min_diff": min_diff
        }
        #print(start, end, cost)
        return cost
    
    def detect_anomalies(self, penalty=5):
        """Esegue il rilevamento delle anomalie e memorizza i change points."""
        algo = rpt.Pelt(custom_cost=self, min_size=self.min_size, jump=1).fit(self.signal)
        self.change_points = algo.predict(pen=penalty)  # Puoi regolare il parametro di penalità
        return self.change_points

    def get_point_explanation(self, idx):
        """
        Restituisce una spiegazione testuale di perché il punto `idx` è stato segnato come anomalo.
        """
        # Trova il segmento a cui appartiene il punto
        for i in range(len(self.change_points) - 1):
            start, end = self.change_points[i], self.change_points[i + 1]
            if start <= idx < end:
                if (start, end) in self.anomaly_details:
                    details = self.anomaly_details[(start, end)]
                    explanation = (
                        f"Punto {idx} è stato rilevato anomalo perché appartiene al segmento [{start}, {end}]:\n"
                        f"- Costo totale: {details['cost']:.4f}\n"
                        f"- Valore KS: {details['ks_value']}\n"
                        f"- Valore EMD: {details['emd_value']}\n"
                        f"- Differenza Media: {details['mean_diff']:.4f}\n"
                        f"- Differenza STD: {details['std_diff']:.4f}\n"
                        f"- Differenza Varianza: {details['var_diff']:.4f}\n"
                        f"- Differenza Max: {details['max_diff']:.4f}\n"
                        f"- Differenza Min: {details['min_diff']:.4f}\n"
                    )
                    return explanation
                else:
                    return f"Punto {idx} è in un segmento [{start}, {end}], ma non presenta un'anomalia evidente."
        return ""

    @property
    def model(self):
        return "kl_stats"