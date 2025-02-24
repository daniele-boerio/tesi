import pandas as pd
from datetime import datetime, timedelta
from GranularData import GranularData
import plotly.graph_objs as go
import plotly.io as pio

measures = ['date_start','date_end','nap','hr_average','hr_max','hr_min','rmssd_average',
 'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min',
 'snoring_average','snoring_max','snoring_min','mvt_score_average','mvt_score_max','mvt_score_min','light_sleep_duration','rem_sleep_duration',
 'deep_sleep_duration','wakeup_duration','wakeup_count','manual_sleep_duration','undefined_sleep_duration',
 'out_of_bed_count','out_of_bed_time','out_of_bed','total_sleep_time','hash_deviceid','model','model_id','date']

class AggregatedFromGranularData:

    #This class accepts a dictionary of nights, where a night is a dictionary of measurements from the previous night to the morning of a specific day of the year.
    def __init__(self, graWithings):
        self.granularData = GranularData(graWithings)
        self.night_data = self.granularData.getDictNights()
        self.measure_dict = generateMeasures(self.night_data)

    #return the granular data class
    def getGranularData(self):
        return self.granularData

    #return the copy of the dictionary of nights
    def getGranularNights(self):
        return self.night_data.copy()
    
    #return the copy of the dictionary with aggregated data of each day of the year
    def getMeasureDict(self):
        return self.measure_dict.copy()
    
    #return the dataframe with aggregated data of each day of the year
    def getMeasureDf(self):
        sleep_data = self.getMeasureDict()
        merged_df = pd.concat(sleep_data.values(), ignore_index=True)
        return merged_df
    
    def getMeasuresOfSpecificNight(self, specific_date):
        # Extract values on a specific data
        measureDict = self.getMeasureDict() 
        if specific_date in measureDict:
            df = pd.DataFrame(measureDict[specific_date])
            return df
        else:
            required_columns = ['date_start','date_end','nap','hr_average','hr_max','hr_min','rmssd_average',
                                'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min',
                                'snoring_average','snoring_max','snoring_min','mvt_score_average','mvt_score_max','mvt_score_min','light_sleep_duration',
                                'rem_sleep_duration','deep_sleep_duration','wakeup_duration','wakeup_count','manual_sleep_duration','undefined_sleep_duration',
                                'out_of_bed_count','out_of_bed_time','out_of_bed','total_sleep_time','hash_deviceid','model','model_id','date']
            return pd.DataFrame(columns=required_columns)
    
    def getMeasuresOfSpecificInterval(self, start_date, end_date):
            nights = self.getMeasureDict()
            # Filtra i DataFrame nell'intervallo di date e concatenali
            filtered_dfs = [
                df for date_str, df in nights.items()
                if start_date <= date_str <= end_date
            ]
            if filtered_dfs:
                # Concatenazione dei DataFrame filtrati
                df_concatenato = pd.concat(filtered_dfs, ignore_index=True)
                return df_concatenato
            else:
                required_columns = ['date_start','date_end','nap','hr_average','hr_max','hr_min','rmssd_average',
                                'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min',
                                'snoring_average','snoring_max','snoring_min','mvt_score_average','mvt_score_max','mvt_score_min','light_sleep_duration',
                                'rem_sleep_duration','deep_sleep_duration','wakeup_duration','wakeup_count','manual_sleep_duration','undefined_sleep_duration',
                                'out_of_bed_count','out_of_bed_time','out_of_bed','total_sleep_time','hash_deviceid','model','model_id','date']
                return pd.DataFrame(columns=required_columns)
    
    def getMeasureOfNights(self, measure):
        # Definire le colonne richieste
        required_columns = ['date', measure]
        df = self.getMeasureDf()
        # Verificare se tutte le colonne richieste sono presenti nel DataFrame
        if all(col in df.columns for col in required_columns):
            # Se tutte le colonne sono presenti, selezionarle e restituire il DataFrame filtrato
            return df[required_columns]
        else:
            # Se manca una delle colonne, restituire un DataFrame vuoto con le colonne richieste
            return 'Measure not present in DF'
    
    def getMeasureOfSpecificNight(self, measure, specific_date):
        required_columns = ['date', measure]
        dict = self.getMeasureDict()
        if specific_date in dict:
            df = dict[specific_date]
            if all(col in df.columns for col in required_columns):
                # Se tutte le colonne sono presenti, selezionarle e restituire il DataFrame filtrato
                if not df.empty:
                    return df[required_columns]
                else:
                    return pd.DataFrame(columns=required_columns)
            else:
                # Se manca una delle colonne, restituire un DataFrame vuoto con le colonne richieste
                return 'Measure not present in DF'
        else:
            return pd.DataFrame(columns=required_columns)
        
    def outOfBedOfSpecificNight(self, specific_date):
        gaps = self.getMeasureOfSpecificNight('out_of_bed', specific_date)['out_of_bed']

        # Lista per raccogliere tutti i dati
        all_data = []
        # Itera attraverso ogni elemento di 'data' e raccogli i dati 'OutOfBed'
        for gap in gaps:
            if isinstance(gap, list):
                    all_data.extend(gap)
        if len(all_data)>0:
            df = pd.DataFrame(all_data)
            return df
        else:
            required_columns = ['date_start','date_end', 'gap_duration']
            return pd.DataFrame(columns=required_columns)
    
    def plot_wake_up_periods(self, start_date, end_date, generate_html=False):
        # Generate a list of dates between start and end date
        date_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')

        fig = go.Figure()

        for idx, specific_date in enumerate(date_list):
            # Obtain the measure of Start and End on a specific date
            sleep_starts = self.getMeasureOfSpecificNight("date_start", specific_date)
            sleep_ends = self.getMeasureOfSpecificNight("date_end", specific_date)

            # Plot sleep time in blue
            for start, end in zip(sleep_starts["date_start"], sleep_ends["date_end"]):
                start_time_obj = pd.to_datetime(start)
                end_time_obj = pd.to_datetime(end)

                # Manage sleep between two dates
                if start_time_obj.date() != end_time_obj.date():
                    # First part, from_time to midnight (23:59)
                    if idx > 0:
                        start_time = start_time_obj.hour + start_time_obj.minute / 60
                        fig.add_trace(go.Scatter(
                            x=[start_time, 24], y=[idx-1, idx-1], mode='lines',
                            line=dict(color='blue', width=7),
                            hovertemplate=f'{start_time_obj.date()} - {start_time_obj.strftime("%H:%M:%S")} - 24:00:00',
                            showlegend=False
                        ))

                    # Second part, from midnight to_time of the current day
                    to_time = end_time_obj.hour + end_time_obj.minute / 60
                    fig.add_trace(go.Scatter(
                        x=[0, to_time], y=[idx, idx], mode='lines',
                        line=dict(color='blue', width=7),
                        hovertemplate=f'{end_time_obj.date()} - 00:00:00 - {end_time_obj.strftime("%H:%M:%S")}',
                        showlegend=False
                    ))
                else:
                    # here i have a sleep in a single day
                    start_time = start_time_obj.hour + start_time_obj.minute / 60
                    end_time = end_time_obj.hour + end_time_obj.minute / 60
                    fig.add_trace(go.Scatter(
                        x=[start_time, end_time], y=[idx, idx], mode='lines',
                        line=dict(color='blue', width=7),
                        hovertemplate=f'{start_time_obj.date()} - {start_time_obj.strftime("%H:%M:%S")} - {end_time_obj.strftime("%H:%M:%S")}',
                        showlegend=False
                    ))

            wake_up_periods = self.outOfBedOfSpecificNight(specific_date)
            for first, second  in zip(wake_up_periods['date_start'], wake_up_periods['date_end']):
                from_time_obj = pd.to_datetime(first)
                to_time_obj =  pd.to_datetime(second)

                #if the last time of the sleep is not in the current date we put the wake up time in the previous day
                if to_time_obj.date() != datetime.strptime(specific_date, "%Y-%m-%d").date():
                    if idx > 0:
                        from_time = from_time_obj.hour + from_time_obj.minute / 60
                        to_time = to_time_obj.hour + to_time_obj.minute / 60
                        fig.add_trace(go.Scatter(
                            x=[from_time, to_time], y=[idx-1, idx-1], mode='lines',
                            line=dict(color='red', width=7),
                            hovertemplate=f'{from_time_obj.date()} - {from_time_obj.strftime("%H:%M:%S")} - {to_time_obj.strftime("%H:%M:%S")}',
                            showlegend=False
                        ))
                #if the first time date of the sleep is not equal to the last time date of the sleep, this mean that a person is wake up across the midnight
                elif from_time_obj.date() != to_time_obj.date():
                    if idx > 0:
                        from_time = from_time_obj.hour + from_time_obj.minute / 60
                        fig.add_trace(go.Scatter(
                            x=[from_time, 24], y=[idx-1, idx-1], mode='lines',
                            line=dict(color='red', width=7),
                            hovertemplate=f'{from_time_obj.date()} - {from_time_obj.strftime("%H:%M:%S")} - 24:00',
                            showlegend=False
                        ))
                    to_time = to_time_obj.hour + to_time_obj.minute / 60
                    fig.add_trace(go.Scatter(
                        x=[0, to_time], y=[idx, idx], mode='lines',
                        line=dict(color='red', width=7),
                        hovertemplate=f'{to_time_obj.date()} - 00:00:00 - {to_time_obj.strftime("%H:%M:%S")}',
                        showlegend=False
                    ))
                #we have a wake up time in the current day so we simply plot it
                else:
                    from_time = from_time_obj.hour + from_time_obj.minute / 60
                    to_time = to_time_obj.hour + to_time_obj.minute / 60
                    fig.add_trace(go.Scatter(
                        x=[from_time, to_time], y=[idx, idx], mode='lines',
                        line=dict(color='red', width=7),
                        hovertemplate=f'{from_time_obj.date()} - {from_time_obj.strftime("%H:%M:%S")} - {to_time_obj.strftime("%H:%M:%S")}',
                        showlegend=False
                    ))
        if len(date_list) <= 30:
            step = 1  # show every day
        elif 30 < len(date_list) <= 90:
            step = 7  # show a day for week
        else:
            step = 30  # show a day for monch
        display_dates = date_list[::step]
        display_indices = list(range(0, len(date_list), step))

        fig.update_layout(
            xaxis=dict(title='Hour of the day', range=[0, 24], tickvals=list(range(0, 25))),
            yaxis=dict(title='Date', tickvals=display_indices, ticktext=display_dates),
            height=800, width=1500,
            showlegend=False,
            title='sleep and wake up time between ' + start_date + ' and ' + end_date
        )
        
        if generate_html:
            start_date, end_date
            file_name = f"AggregatedFromGranular_wakeUpPeriods_{start_date}_{end_date}"
            pio.write_html(fig, file='graphs/'+file_name+'.html', auto_open=True)

        fig.show()

    def plot(self, measure, start_date, end_date, generate_html=False):
        # Generate a list of dates between start and end date
        date_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')

        dates = []
        values = []

        for specific_date in date_list:
            # Obtain the measure on a specific date
            sleepsMeasure = self.getMeasureOfSpecificNight(measure, specific_date)

            # Extract the values from the list of measures
            if len(sleepsMeasure) > 0:
                specific_values = sleepsMeasure[measure].tolist()
                values.append(specific_values)
                dates.append(specific_date)

        fig = go.Figure()

        for idx, value_list in enumerate(values):
            for value in value_list:
                fig.add_trace(go.Scatter(
                    x=[dates[idx]],
                    y=[value],
                    mode='markers',
                    marker=dict(size=7, color='Blue'),
                    name=f'{measure} ({dates[idx]})',
                    text=f'{measure}: {value}'
                ))

        fig.update_layout(
            title=f'Valori di {measure} da {start_date} a {end_date}',
            xaxis_title='Date', 
            yaxis_title=f'Valori di {measure}',
            yaxis={'categoryorder': 'category ascending'}, 
            xaxis={'categoryorder': 'category ascending'},
            height=800, width=1500,
            showlegend=False,
        )

        if generate_html:
            file_name = f"AggregatedFromGranular_{measure}_{start_date}_{end_date}"
            pio.write_html(fig, file='graphs/'+file_name+'.html', auto_open=True)

        fig.show()



def generateMeasures(night_data):
    aggregated_dict = {}
    sleep_values = pd.DataFrame()
    df = pd.DataFrame()
    merge_dict = {}
    
    for date, sleepDFs in night_data.items():
        for sleep in sleepDFs:
            sleep_statistics_dict = calculateSleepStatistics(sleep)
            heart_statistics_dict = calculateHeartStatistics(sleep)
            out_of_bed_dict = calculateOutOfBedStatistics(sleep)

            merge_dict = sleep_statistics_dict.copy()
            merge_dict.update(heart_statistics_dict)
            merge_dict.update(out_of_bed_dict)

            LightSleepDuration = merge_dict['light_sleep_duration']
            DeepSleepDuration = merge_dict['deep_sleep_duration']
            RemSleepDuration = merge_dict['rem_sleep_duration']
            ManualSleepDuration = merge_dict['manual_sleep_duration']
            UndefinedSleepDuration = merge_dict['undefined_sleep_duration']
            merge_dict['total_sleep_time'] = LightSleepDuration + DeepSleepDuration + RemSleepDuration + ManualSleepDuration + UndefinedSleepDuration
            if len(sleep['model']) > 0:
                merge_dict['model'] = sleep['model'].iloc[0]
            else:
                merge_dict['model'] = None
            if len(sleep['model_id']) > 0:
                merge_dict['model_id'] = sleep['model_id'].iloc[0]
            else:
                merge_dict['model_id'] = None
            merge_dict['date'] = date

            df = pd.DataFrame([merge_dict])
            sleep_values = pd.concat([sleep_values, df], ignore_index=True)
        aggregated_dict[date] = sleep_values
        sleep_values = pd.DataFrame()

    return aggregated_dict


def calculateSleepStatistics(sleep):
    values_dict = {}
    startValue = None
    endValue = None
    nap = False

    if not sleep.empty:
        # Take the first value of the StartValue and last value of EndValue
        startValue = sleep['local_time'].iloc[0]
        endValue = sleep['local_time'].iloc[-1]

    if startValue is not None and endValue is not None:
        startDay = pd.to_datetime(startValue).day
        endDay = pd.to_datetime(endValue).day
        startHour = pd.to_datetime(startValue).hour
        endHour = pd.to_datetime(endValue).hour

        #if the sleep start after 10:00 and concludes before 23:00 is a nap
        if(startDay == endDay):
            if (startHour > 10 and endHour < 23):
                nap = True

        values_dict = {
            "date_start": startValue,
            "date_end": endValue,
            "nap": nap
        }

    return values_dict


def calculateHeartStatistics(sleep):
    # Inizializza il dizionario aggregato
    aggregated_dict = {}

    if not sleep.empty:
        # Rimuovi eventuali valori NaN per le colonne numeriche
        numeric_cols = ['hr', 'rmssd', 'rr', 'sdnn_1', 'snoring', 'mvt_score']
        stats = sleep[numeric_cols]

        # Calcola statistiche per ogni colonna numerica
        aggregated_stats = stats.agg(['max', 'min', 'mean']).to_dict()

        # Converte le statistiche in valori singoli
        for col in numeric_cols:
            aggregated_dict[f"{col}_average"] = int(aggregated_stats[col]['mean']) if pd.notna(aggregated_stats[col]['mean']) else None
            aggregated_dict[f"{col}_max"] = int(aggregated_stats[col]['max']) if pd.notna(aggregated_stats[col]['max']) else None
            aggregated_dict[f"{col}_min"] = int(aggregated_stats[col]['min']) if pd.notna(aggregated_stats[col]['min']) else None

        # Gestisci il tempo trascorso in ogni stato
        sleep['time_diff'] = sleep['local_time'].diff().fillna(timedelta(0))
        sleep['time_diff'] = sleep['time_diff'].dt.total_seconds().astype(int)

        time_in_states = sleep.groupby('state')['time_diff'].sum().to_dict()
        wakeup_count = ((sleep['state'].shift(1) != 0) & (sleep['state'] == 0)).sum()

        # Popola il dizionario per i tempi di stato
        aggregated_dict["light_sleep_duration"] = time_in_states.get(1, 0)
        aggregated_dict["rem_sleep_duration"] = time_in_states.get(3, 0)
        aggregated_dict["deep_sleep_duration"] = time_in_states.get(2, 0)
        aggregated_dict["wakeup_duration"] = time_in_states.get(0, 0)
        aggregated_dict["manual_sleep_duration"] = time_in_states.get(4, 0)
        aggregated_dict["undefined_sleep_duration"] = time_in_states.get(5, 0)
        aggregated_dict["wakeup_count"] = wakeup_count

    return aggregated_dict



def calculateOutOfBedStatistics(sleep):         
    out_of_bed_dict = {}
    
    # Trova e filtra i gap fuori dal letto
    out_of_bed_values = find_and_filter_gaps_to_zero(sleep)
    
    # Conta il numero di gap fuori dal letto
    out_of_bed_count = len(out_of_bed_values)
    
    # Inizializza una lista per contenere i timedeltas
    timedeltas = []

    # Cicla sulle righe del DataFrame
    for _, row in out_of_bed_values.iterrows():
        timedeltas.append(pd.to_timedelta(row['gap_duration']))

    # Somma tutte le durate
    out_of_bed_time = sum(timedeltas, timedelta())

    # Verifica se il tempo fuori dal letto è 0 e imposta un formato appropriato
    if out_of_bed_time == timedelta(0):
        out_of_bed_time = pd.Timedelta(days=0, hours=0, minutes=0)

    # Crea il dizionario con i risultati
    out_of_bed_dict = {
        "out_of_bed_count": out_of_bed_count,
        "out_of_bed_time": out_of_bed_time,
        "out_of_bed": out_of_bed_values.to_dict(orient='records')  # Converti il DataFrame in una lista di dizionari
    }

    return out_of_bed_dict

def find_and_filter_gaps_to_zero(sleep, time_threshold=5 * 60):
    # Filtra solo le righe con valori non nulli in 'state' e ordina per 'local_time'
    filtered_df = sleep[sleep['state'].notnull()].sort_values(by='local_time')

    # Calcola le differenze temporali (in secondi) tra righe consecutive
    filtered_df['time_diff'] = filtered_df['local_time'].diff().dt.total_seconds()

    # Identifica i gap di tempo che superano la soglia
    condition_gap = filtered_df['time_diff'] > time_threshold

    # Filtra solo le righe che soddisfano la condizione
    gaps = filtered_df[condition_gap].copy()

    # Controlla se il DataFrame è vuoto
    if not gaps.empty:
        # Ottieni informazioni di stato precedente, corrente e successivo
        gaps['from_value'] = filtered_df['state'].shift(1)
        gaps['to_value'] = filtered_df['state']
        gaps['next_value'] = filtered_df['state'].shift(-1)
        gaps['date_start'] = filtered_df['local_time'].shift(1)
        gaps['date_end'] = filtered_df['local_time']

        # Filtra i gap che vanno da uno stato diverso da 0 a 0, e il valore successivo è ancora 0
        gaps_to_zero = gaps[(gaps['from_value'] != 0) & (gaps['to_value'] == 0) & (gaps['next_value'] == 0)].copy()

        gaps_to_zero['gap_duration'] = gaps_to_zero['time_diff'].apply(
            lambda x: pd.Timedelta(seconds=x) if pd.notna(x) else timedelta(0)
        )

        # Crea la lista finale dei risultati con date di inizio (from_time) e fine (to_time)
        filtered_time_gaps = gaps_to_zero.loc[:, ['date_start', 'date_end', 'gap_duration']]
        return filtered_time_gaps
    return pd.DataFrame()

