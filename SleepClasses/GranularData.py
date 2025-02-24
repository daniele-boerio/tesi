import pandas as pd
import plotly.graph_objs as go
import json
import plotly.io as pio
import plotly.express as px

#measures = ["timestamp (GMT)", "GMT_time", "local_time", "hr", "rmssd", "rr", "sdnn_1", "snoring", "state", "mvt_score", "hash_deviceid", "model", "model_id"]

class GranularData:
    # This class accepts a dictionary of dataframes for a specific subject
    def __init__(self, sleep_df):
        self.sleep_df = sleep_df
        self.nights = divideInNights(self.sleep_df)
    
    def getDataCleaned(self):
        return self.sleep_df.copy()
    
    def getDictNights(self):
        return self.nights.copy()
    
    def getMeasuresOfSpecificNight(self, specific_date):
        # Extract values on a specific data
        nights = self.getDictNights()
        if specific_date in nights:
            return pd.concat(nights[specific_date], ignore_index=True)
        else:
            required_columns = ["timestamp (GMT)", "GMT_time", "local_time", "hr", "rmssd", "rr", "sdnn_1", "snoring", "state", "mvt_score", "hash_deviceid", "model", "model_id"]
            return pd.DataFrame(columns=required_columns)
    
    def getMeasuresOfSpecificInterval(self, start_time, end_time):
        # Crea un DataFrame vuoto per accumulare i risultati
        nights = self.getDictNights()
        result_df = pd.DataFrame()

        start_day = pd.to_datetime(start_time).date()
        end_day = pd.to_datetime(end_time).date()

        filtered_dict = {date_key: dfs for date_key, dfs in nights.items()
            if start_day <= pd.to_datetime(date_key).date() <= end_day}

        # Cicla sui DataFrame nel dizionario
        for date_key, dfs in filtered_dict.items():
            for df in dfs:
                # Filtra solo le righe in cui la colonna 'Date' è tra start_time ed end_time
                mask = (pd.to_datetime(df['local_time']) >= start_time) & (pd.to_datetime(df['local_time']) <= end_time)
                filtered_df = df[mask]

                # Aggiungi il DataFrame filtrato al risultato
                result_df = pd.concat([result_df, filtered_df], ignore_index=True)

        return result_df
        
            
    def getMeasureOfNights(self, measure):
        # Definire le colonne richieste
        required_columns = ['local_time', measure]
        df = self.getDataCleaned()
        # Verificare se tutte le colonne richieste sono presenti nel DataFrame
        if all(col in df.columns for col in required_columns):
            # Se tutte le colonne sono presenti, selezionarle e restituire il DataFrame filtrato
            return df[required_columns]
        else:
            # Se manca una delle colonne, restituire un DataFrame vuoto con le colonne richieste
            return 'Measure not present in DF'
            
    
    def getMeasureOfSpecificNight(self, measure, specific_date):
        required_columns = ['local_time', measure]
        sleeps = self.getDictNights()
        if specific_date in sleeps:

            result = pd.concat(sleeps[specific_date], ignore_index=True)
            if all(col in result.columns for col in required_columns):
                # Se tutte le colonne sono presenti, selezionarle e restituire il DataFrame filtrato
                if not result.empty:
                    return result[required_columns]
                else:
                    return pd.DataFrame(columns=required_columns)
            else:
                return 'Measure not present in DF'
        else:
            return pd.DataFrame(columns=required_columns)
    

    def plot(self, measure, day, generate_html=False):
        nights = self.getDictNights()
        
        if day not in nights:
            return f"No data available for {day}"
        
        # for a specific measure we retrieve the specific measure of the specific day
        match measure:
            case 'state':
                sleepDF = self.getMeasureOfSpecificNight('state', day)
                title = 'State '
            case 'snoring':
                sleepDF = self.getMeasureOfSpecificNight('snoring', day)
                title = 'Snoring '
            case 'hr':
                sleepDF = self.getMeasureOfSpecificNight('hr', day)
                title = 'Heart Rate '
            case 'rmssd':
                sleepDF = self.getMeasureOfSpecificNight('rmssd', day)
                title = 'Root mean square of the successive differences Heart Rate variability '
            case 'rr':
                sleepDF = self.getMeasureOfSpecificNight('rr', day)
                title = 'Respiration Rate '
            case 'sdnn_1':
                sleepDF = self.getMeasureOfSpecificNight('sdnn_1', day)
                title = 'Standard deviation of the Normal to Normal Heart Rate variability '
            case 'mvt_score':
                sleepDF = self.getMeasureOfSpecificNight('mvt_score', day)
                title = 'Intensity of movement in bed on a minute-by-minute basis '
            case _:
                print("Measure not valid")
                return
        
        if(not sleepDF.empty):
            df = pd.DataFrame(sleepDF)
        else:
            return 'Measure not present in DF'
        # Creare il grafico scatter con Plotly
        fig = px.scatter(
            sleepDF, 
            x='local_time',
            y=f'{measure}',
            title=title,
            labels={'Timestamp': 'Date and Hours', 'Value': measure}
        )
        fig.update_layout(xaxis_title="Date and Hours", yaxis_title="Value")
        
        if generate_html:
            # Save the graph on a file
            file_name = f"granularData_{measure}_{day}"
            pio.write_html(fig, file='graphs/'+file_name+'.html', auto_open=True)
        
        fig.show()

# Function for discover duplicates and remove rows if one is subset of another one
def remove_subset_json(df):
    duplicates = df[df.duplicated(subset=['local_time'], keep=False)]
    
    # List of indexes to delete
    indices_to_drop = []

    for time, group in duplicates.groupby('local_time'):
        # Convert strings of JSON in dictionary
        json_dicts = [json.loads(val) for val in group['Value']]
        
        # Check if one JSON is a subset of another one
        for i in range(len(json_dicts)):
            for j in range(i + 1, len(json_dicts)):
                if json_dicts[i].items() <= json_dicts[j].items():
                    # Add the row index to the remove list
                    indices_to_drop.append(group.index[i])
                elif json_dicts[j].items() <= json_dicts[i].items():
                    # Add the row index to the remove list
                    indices_to_drop.append(group.index[j])
    
    # Remove rows from the df
    df_cleaned = df.drop(indices_to_drop).reset_index(drop=True)
    return df_cleaned

def divideInNights(df):
    # Ordina il DataFrame per 'local_time'
    df = df.sort_values(by='local_time').reset_index(drop=True)
    
    # Converte la colonna 'local_time' in datetime (se non già fatto)
    df['local_time'] = pd.to_datetime(df['local_time'], utc=True)
    df['local_time'] = df['local_time'].dt.tz_convert('Europe/Rome')
    
    # Calcola la differenza tra la riga corrente e la precedente, partendo dalla riga 1
    df['time_diff'] = df['local_time'] - df['local_time'].shift(1)
    # Riempie i NaT con 0 per la differenza di tempo
    df['time_diff'] = df['time_diff'].fillna(pd.Timedelta(seconds=0))
    
    # Identifica nuovi segmenti (gap > 2 ore)
    df['new_session'] = (df['time_diff'] > pd.Timedelta(hours=2)).cumsum()
    
    # Raggruppa per 'new_session' per identificare sessioni distinte
    grouped = df.groupby('new_session')
    
    # Inizializza il dizionario per salvare i risultati
    night_data = {}

    for session_id, group in grouped:
        # Determina il giorno corretto: basato sull'ultimo local_time della sessione
        last_local_time = group['local_time'].iloc[-1]
        date_key = last_local_time.date()
        
        # Converti date_key in stringa
        date_key_str = date_key.strftime('%Y-%m-%d')
        
        # Rimuovi le colonne di supporto (se non necessarie)
        group = group.drop(columns=['time_diff', 'new_session'])
        
        # Aggiungi la sessione al dizionario
        if date_key_str not in night_data:
            night_data[date_key_str] = []
        night_data[date_key_str].append(group)

    return night_data