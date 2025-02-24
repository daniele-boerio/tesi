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

class AnomalyDetection:

    #This class accepts...
    def __init__(self, df, windowSize, overlapWindow):
        self.df = df
        self.windowSize = windowSize
        self.overlapWindow = overlapWindow
        # Registra la funzione di costo personalizzata in ruptures
        rpt.costs.CostGroundTruth = CostGroundTruth
    
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

    def start(self, column, penalty, minSize, aggregatedDays=-1, percentChangePoints = 0.5):
        if aggregatedDays == -1:
            aggregatedDays = minSize
        # Estrazione dei dati per l'analisi
        signal = np.vstack([self.df[column].values]).T  # Converti in array 2D (richiesto da ruptures)
        
        # Creazione del MinMaxScaler per scalare i dati tra -1 e 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # Adattamento e trasformazione del segnale
        signal_scaled = scaler.fit_transform(signal)
        
        # Inizializzazione delle variabili
        start_idx = 0
        end_idx = self.windowSize
        breakpoints_dict = {}

        overlap = self.windowSize - self.overlapWindow

        while end_idx <= len(signal_scaled):
            # Estrazione della ground truth e del segnale da controllare
            groundTruth = signal_scaled[start_idx:end_idx]
            signalToCheck = signal_scaled[start_idx + overlap + 1 :]

            if len(signalToCheck) < minSize:
                break  # Oppure continua con un'altra gestione

            # Creazione del modello PELT con la funzione di costo personalizzata
            cost = rpt.costs.CostGroundTruth(groundTruth=groundTruth, minSize=minSize)
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
            x=np.arange(len(signal_scaled)), 
            y=signal_scaled.flatten(), 
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

    def startTrend(self, column, penalty, minSize, aggregatedDays=-1, percentChangePoints = 0.5):
        if aggregatedDays == -1:
            aggregatedDays = minSize
        # Estrazione dei dati per l'analisi
        signal = np.vstack([self.df[column].values]).T  # Converti in array 2D (richiesto da ruptures)
        
        # Creazione del MinMaxScaler per scalare i dati tra -1 e 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # Adattamento e trasformazione del segnale
        signal_scaled = scaler.fit_transform(signal)

        decomposed = seasonal_decompose(signal_scaled, model='additive', period=7)
        trend = pd.Series(decomposed.trend).dropna()
        # Scala il trend estratto
        #trend_scaled = scaler.fit_transform(trend.values.reshape(-1, 1)).flatten()  # Scala il trend
        
        # Inizializzazione delle variabili
        start_idx = 0
        end_idx = self.windowSize
        breakpoints_dict = {}

        overlap = self.windowSize - self.overlapWindow

        while end_idx <= len(signal_scaled):
            # Estrazione della ground truth e del segnale da controllare
            groundTruth = trend[start_idx:end_idx]
            signalToCheck = trend[start_idx + overlap + 1 :]

            if len(signalToCheck) < minSize:
                break  # Oppure continua con un'altra gestione

            # Creazione del modello PELT con la funzione di costo personalizzata (residui)
            cost = rpt.costs.CostGroundTruth(groundTruth=groundTruth, minSize=minSize)
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

            # Aggiungi altri breakpoint finché sono all'interno della finestra di aggregazione
            while sorted_keys and sorted_keys[0] <= ref_bp + aggregatedDays:
                to_merge.append(sorted_keys.pop(0))
            
            # Prendi il primo punto con valore massimo
            max_bp = max(to_merge, key=lambda bp: breakpoints_dict[bp])
            
            # Aggiungi il punto con il valore massimo
            merged_breakpoints[max_bp] = sum(breakpoints_dict[bp] for bp in to_merge)

        final_breakpoints = {}
        for key, value in merged_breakpoints.items():
            if(key // overlap * percentChangePoints < value):
                final_breakpoints[key] = value
        
        # Crea il grafico con Plotly
        fig = go.Figure()

        # Traccia il segnale scalato
        fig.add_trace(go.Scatter(
            x=np.arange(len(trend)), 
            y=trend, 
            mode='lines', 
            name='Trend Scaled'
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
            title=f"Change Points on Trend {column}",
            xaxis_title="Day",
            yaxis_title="Time",
            #template="plotly_dark",
            showlegend=True
        )

        # Mostra il grafico
        fig.show()
        
        # Crea il grafico con Plotly
        fig = go.Figure()

        # Traccia il segnale scalato
        fig.add_trace(go.Scatter(
            x=np.arange(len(signal_scaled)), 
            y=signal_scaled.flatten(), 
            mode='lines', 
            name='Signal Scaled'
        ))

        for breakpoint in final_breakpoints.keys():
            # Aggiungi una linea verticale a x = 50
            fig.add_shape(
                type="line",
                x0=breakpoint, x1=breakpoint,  # Linea fissa a x = 50
                y0=-1, y1=1,  # Estende la linea sull'asse Y (da 0 a 1)
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
        #rpt.display(trend, final_breakpoints.keys())
        #rpt.display(signal_scaled, final_breakpoints.keys())
        #plt.xlabel("Day")
        #plt.ylabel("Time")
        #plt.show()

    def startSeasonality(self, column, penalty, minSize, aggregatedDays=7, percentChangePoints = 0.5):
        if aggregatedDays == -1:
            aggregatedDays = minSize
        # Estrazione dei dati per l'analisi
        signal = np.vstack([self.df[column].values]).T  # Converti in array 2D (richiesto da ruptures)
        
        # Creazione del MinMaxScaler per scalare i dati tra -1 e 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # Adattamento e trasformazione del segnale
        signal_scaled = scaler.fit_transform(signal)

        decomposed = seasonal_decompose(signal_scaled, model='additive', period=30)
        seasonality = pd.Series(decomposed.seasonal).dropna()
        
        # Inizializzazione delle variabili
        start_idx = 0
        end_idx = self.windowSize
        breakpoints_dict = {}

        overlap = self.windowSize - self.overlapWindow

        while end_idx <= len(signal_scaled):
            # Estrazione della ground truth e del segnale da controllare
            groundTruth = seasonality[start_idx:end_idx]
            signalToCheck = seasonality[start_idx + overlap + 1 :]

            if len(signalToCheck) < minSize:
                break  # Oppure continua con un'altra gestione

            # Creazione del modello PELT con la funzione di costo personalizzata (residui)
            cost = rpt.costs.CostGroundTruth(groundTruth=groundTruth, minSize=minSize)
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
        
        # Visualizza il grafico
        plt.figure(figsize=(10, 6))
        rpt.display(seasonality, final_breakpoints.keys())
        rpt.display(signal_scaled, final_breakpoints.keys())
        plt.xlabel("Day")
        plt.ylabel("Time")
        plt.title("Change points on seasonality")
        plt.show()

class CostGroundTruth(BaseCost):
    """
    Funzione di costo personalizzata per il confronto tra segmenti
    del segnale e la ground truth utilizzando la Divergenza di Kullback-Leibler.
    """
    model = "custom_ground_truth"
    
    def __init__(self, groundTruth, minSize, epsilon=1e-10):
        super().__init__()
        self.groundTruth = groundTruth
        self.min_size = minSize
        self.epsilon = epsilon  # Costante di regolarizzazione per evitare log(0) e divisione per 0

        # Parametri della distribuzione della ground truth
        self.mean_groundTruth = np.mean(groundTruth, axis=0)  # Media della ground truth
        self.std_groundTruth = np.std(groundTruth, axis=0)    # Deviazione standard della ground truth

    def fit(self, signal):
        """
        Imposta il segnale da analizzare.
        """
        self.signal = signal
        return self

    def kl_divergence(self, p, q):
        """
        Calcola la Divergenza di Kullback-Leibler tra due distribuzioni p e q.
        p è la distribuzione della ground truth e q è la distribuzione del segmento.
        """
        # Aggiungere un epsilon per evitare divisioni per zero
        p = np.maximum(p, self.epsilon)
        q = np.maximum(q, self.epsilon)
        
        kl_div = np.sum(p * np.log(p / q), axis=0)  # Calcola la KL divergence
        return kl_div
    
    def error(self, start, end):
        segment_length = end - start
        if segment_length < self.min_size:
            return  # Scarta segmenti troppo piccoli

        segment = self.signal[start:end]
        mean_segment = np.mean(segment, axis=0)
        std_segment = np.std(segment, axis=0)

        # Calcolo distribuzioni
        p = norm.pdf(segment, loc=self.mean_groundTruth, scale=self.std_groundTruth)
        q = norm.pdf(segment, loc=mean_segment, scale=std_segment)

        kl_div = self.kl_divergence(p, q)

        # Penalità per i segmenti troppo corti (graduale)
        segment_length_penalty = np.exp(-0.5 * (self.min_size - (end - start))**2)  # Penalità esponenziale per lunghezza

        # Penalità temporale per segmenti lunghi
        time_penalty = np.exp(-0.5 * segment_length)

        # Il costo totale è la divergenza KL più la penalità sulla lunghezza e sulla durata temporale
        total_error = kl_div + segment_length_penalty + time_penalty

        return total_error