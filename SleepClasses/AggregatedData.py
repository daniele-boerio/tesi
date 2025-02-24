import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from statsmodels.tsa.seasonal import seasonal_decompose
from AggregatedFromGranularData import AggregatedFromGranularData
from AggregatedDataFromWithings import AggregatedDataFromWithings

#measures = ['date','date_start','date_end','light_sleep_duration','deep_sleep_duration','rem_sleep_duration','wakeup_duration',
# 'total_sleep_time','num_rem_episodes','sleep_efficiency','sleep_latency','total_time_in_bed','apnea_hypopnea_index','breathing_disturbances_intensity',
# 'asleep_duration','mvt_active_duration','night_events','snoring','wakeup_latency','waso','snoring_episode_count','sleep_score','sleep_count',
# 'hash_device_id','model','model_id','manual_sleep_duration','undefined_sleep_duration','nap','hr_average','hr_max','hr_min','rmssd_average',
# 'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min','snoring_average','snoring_max','snoring_min',
# 'mvt_score_average','mvt_score_max','mvt_score_min','wakeup_count','out_of_bed_count','out_of_bed_time','out_of_bed']


class AggregatedData:

    #This class accepts...
    def __init__(self, graWithings, aggWithings):
        self.aggregatedGranular = AggregatedFromGranularData(graWithings)
        self.aggregatedWithings = AggregatedDataFromWithings(aggWithings)
        self.final_dict, self.modified_rows_df, self.zeros_rows_df = mergeAggregatedData(self.aggregatedGranular, self.aggregatedWithings)
    
    #return the aggregated Data class from Granular Data 
    def getAggregatedDataFromGranular(self):
        return self.aggregatedGranular
    
    #return the aggregated Data class from Granular Data 
    def getAggregatedDataFromWithings(self):
        return self.aggregatedWithings
    
    #return the merged dictionary of aggregated data 
    def getAggregatedDataDict(self):
        return self.final_dict.copy()
    
    #return the dataframe with aggregated data of each day of the year
    def getAggregatedDataDf(self):
        sleep_data = self.getAggregatedDataDict()

        # Controlla se il dizionario è vuoto
        if not sleep_data:  # Il dizionario è vuoto
            required_columns = ['date','date_start','date_end','light_sleep_duration','deep_sleep_duration','rem_sleep_duration','wakeup_duration',
            'total_sleep_time','num_rem_episodes','sleep_efficiency','sleep_latency','total_time_in_bed','apnea_hypopnea_index','breathing_disturbances_intensity',
            'asleep_duration','mvt_active_duration','night_events','snoring','wakeup_latency','waso','snoring_episode_count','sleep_score','sleep_count',
            'hash_device_id','model','model_id','manual_sleep_duration','undefined_sleep_duration','nap','hr_average','hr_max','hr_min','rmssd_average',
            'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min','snoring_average','snoring_max','snoring_min',
            'mvt_score_average','mvt_score_max','mvt_score_min','wakeup_count','out_of_bed_count','out_of_bed_time','out_of_bed']
            return pd.DataFrame(columns=required_columns)

        # Concatena i DataFrame solo se il dizionario non è vuoto
        merged_df = pd.concat(sleep_data.values(), ignore_index=True)
        return merged_df
    
    def getModifiedRowsDF(self):
        return self.modified_rows_df.copy()
    
    def getRemovedRowsDF(self):
        return self.zeros_rows_df.copy()
    
    def getMeasuresOfSpecificNight(self, specific_date):
        # Extract values on a specific data
        measureDict = self.getAggregatedDataDict() 
        if specific_date in measureDict:
            df = pd.DataFrame(measureDict[specific_date])
            #df = df[df['Nap'] == False]
            return df
        else:
            required_columns = ['date','date_start','date_end','light_sleep_duration','deep_sleep_duration','rem_sleep_duration','wakeup_duration',
            'total_sleep_time','num_rem_episodes','sleep_efficiency','sleep_latency','total_time_in_bed','apnea_hypopnea_index','breathing_disturbances_intensity',
            'asleep_duration','mvt_active_duration','night_events','snoring','wakeup_latency','waso','snoring_episode_count','sleep_score','sleep_count',
            'hash_device_id','model','model_id','manual_sleep_duration','undefined_sleep_duration','nap','hr_average','hr_max','hr_min','rmssd_average',
            'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min','snoring_average','snoring_max','snoring_min',
            'mvt_score_average','mvt_score_max','mvt_score_min','wakeup_count','out_of_bed_count','out_of_bed_time','out_of_bed']
            return pd.DataFrame(columns=required_columns)
    
    def getMeasuresOfSpecificInterval(self, start_date, end_date):
        nights = self.getAggregatedDataDict()
        # Filtra i DataFrame nell'intervallo di date e concatenali
        filtered_dfs = [
            df for date_str, df in nights.items()
            if start_date <= date_str <= end_date
        ]
        # Controlla se il dizionario è vuoto
        if not filtered_dfs:  # Il dizionario è vuoto
            required_columns = ['date','date_start','date_end','light_sleep_duration','deep_sleep_duration','rem_sleep_duration','wakeup_duration',
            'total_sleep_time','num_rem_episodes','sleep_efficiency','sleep_latency','total_time_in_bed','apnea_hypopnea_index','breathing_disturbances_intensity',
            'asleep_duration','mvt_active_duration','night_events','snoring','wakeup_latency','waso','snoring_episode_count','sleep_score','sleep_count',
            'hash_device_id','model','model_id','manual_sleep_duration','undefined_sleep_duration','nap','hr_average','hr_max','hr_min','rmssd_average',
            'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min','snoring_average','snoring_max','snoring_min',
            'mvt_score_average','mvt_score_max','mvt_score_min','wakeup_count','out_of_bed_count','out_of_bed_time','out_of_bed']
            return pd.DataFrame(columns=required_columns)
        # Concatenazione dei DataFrame filtrati
        df_concatenato = pd.concat(filtered_dfs, ignore_index=True)
        if len(df_concatenato)>0:
            return df_concatenato
        else:
            required_columns = ['date','date_start','date_end','light_sleep_duration','deep_sleep_duration','rem_sleep_duration','wakeup_duration',
            'total_sleep_time','num_rem_episodes','sleep_efficiency','sleep_latency','total_time_in_bed','apnea_hypopnea_index','breathing_disturbances_intensity',
            'asleep_duration','mvt_active_duration','night_events','snoring','wakeup_latency','waso','snoring_episode_count','sleep_score','sleep_count',
            'hash_device_id','model','model_id','manual_sleep_duration','undefined_sleep_duration','nap','hr_average','hr_max','hr_min','rmssd_average',
            'rmssd_max','rmssd_min','rr_average','rr_max','rr_min','sdnn_1_average','sdnn_1_max','sdnn_1_min','snoring_average','snoring_max','snoring_min',
            'mvt_score_average','mvt_score_max','mvt_score_min','wakeup_count','out_of_bed_count','out_of_bed_time','out_of_bed']
            return pd.DataFrame(columns=required_columns)
    
    def getMeasureOfNights(self, measure):
        # Definire le colonne richieste
        required_columns = ['date', measure]
        df = self.getAggregatedDataDf()
        # Verificare se tutte le colonne richieste sono presenti nel DataFrame
        if all(col in df.columns for col in required_columns):
            # Se tutte le colonne sono presenti, selezionarle e restituire il DataFrame filtrato
            return df[required_columns]
        else:
            # Se manca una delle colonne, restituire un DataFrame vuoto con le colonne richieste
            return 'Measure not present in DF'
    
    def getMeasureOfSpecificNight(self, measure, specific_date):
        required_columns = ['date', measure]
        dict = self.getAggregatedDataDict()
        if specific_date in dict:
            df = dict[specific_date]
            #df = df[df['Nap']==False]
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
        if gaps is not None:
            for gap in gaps:
                if isinstance(gap, list):
                    all_data.extend(gap)
            if len(all_data)>0:
                return pd.DataFrame(all_data)
        required_columns = ['date_start','date_end','gap_duration']
        return pd.DataFrame(columns=required_columns)
    
    def plot_wake_up_periods(self, start_date, end_date, generate_html=False, save_png= False):
        # Generate a list of dates between start and end date
        date_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')

        fig = go.Figure()

        for idx, specific_date in enumerate(date_list):
            # Obtain the measure of Start and End on a specific date
            sleep_starts = self.getMeasureOfSpecificNight("date_start", specific_date)
            sleep_ends = self.getMeasureOfSpecificNight("date_end", specific_date)

            # Plot sleep time in blue
            for start, end in zip(sleep_starts['date_start'], sleep_ends['date_end']):
                start_time_obj = start
                end_time_obj = end

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
                from_time_obj = first
                to_time_obj = second
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

        if save_png:
            file_name = f"WakeUpPeriods_{start_date}_{end_date}.png"
            full_path = f"../graphs/{file_name}"
            fig.write_image(full_path, format="png", width=1000, height=600)
            print(f"Grafico salvato come PNG in: {full_path}")

        fig.show()

    def plot(self, measure, start_date, end_date, period = 7, generate_html=False, save_png= False):
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
        
        # Flatten values for decomposition (1D array required)
        flattened_values = [item for sublist in values for item in sublist]
        flattened_dates = [dates[idx] for idx, sublist in enumerate(values) for _ in sublist]

        # Ensure data is a time series for decomposition
        ts = pd.Series(data=flattened_values, index=pd.to_datetime(flattened_dates))

        decomposed = seasonal_decompose(ts, model='additive', period= period)
        trend = decomposed.trend  # Exclude NaN values at boundaries

        fig = go.Figure()

        # Add trend line
        fig.add_trace(go.Scatter(
            x=trend.index.strftime('%Y-%m-%d'),
            y=trend.values,
            mode='lines',
            line=dict(color='Red', width=2),
            name='Trend'
        ))

        # Create a single Scatter trace
        fig.add_trace(go.Scatter(
            x=flattened_dates,
            y=flattened_values,
            mode='markers',
            marker=dict(size=7, color='Blue'),
            name=f'{measure}',
            text=[f'{measure}: {value}' for value in flattened_values]
        ))

        fig.update_layout(
            title=f'Valori di {measure} da {start_date} a {end_date}',
            xaxis_title='Date', 
            yaxis_title=f'Valori di {measure}',
            yaxis={'categoryorder': 'category ascending'}, 
            xaxis={'categoryorder': 'category ascending'},
            height=800, width=1500,
            showlegend=True,
        )

        if generate_html:
            file_name = f"AggregatedData_{measure}_{start_date}_{end_date}"
            pio.write_html(fig, file='graphs/'+file_name+'.html', auto_open=True)
        
        if save_png:
            file_name = f"{measure}_{start_date}_{end_date}.png"
            full_path = f"../graphs/{file_name}"
            fig.write_image(full_path, format="png", width=1000, height=600)
            print(f"Grafico salvato come PNG in: {full_path}")

        fig.show()

    def plot_AggregatedData(self,measure, start_date, end_date, group_by="month", generate_html=False, save_png= False):
        
        # Genera una lista di date tra start_date ed end_date
        date_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')
        dates = []
        values = []

        for specific_date in date_list:
            # Ottieni la misura per una specifica notte
            sleepsMeasure = self.getMeasureOfSpecificNight(measure, specific_date)
            if len(sleepsMeasure) > 0:
                # Estrai i valori dalla lista di misure
                specific_values = sleepsMeasure[measure].tolist()
                values.extend(specific_values)  # Aggiungi tutti i valori (può essere una lista)
                dates.extend([specific_date] * len(specific_values))  # Ripeti la data per ogni valore

        # Controlla se ci sono dati sufficienti
        if len(dates) == 0 or len(values) == 0:
            print("Nessun dato disponibile per il periodo selezionato.")
            return
        if(measure == 'total_sleep_time'):
            
            df = pd.DataFrame({"date": pd.to_datetime(dates), "value": pd.to_timedelta(values, unit="s")})
        else:
            df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})

        # Calcola la media giornaliera per ciascun giorno
        daily_stats = df.groupby("date")["value"].sum().reset_index()

        # Raggruppa per settimana o mese
        if group_by == "month":
            daily_stats["period"] = daily_stats["date"].dt.to_period("M")  # Raggruppa per mese
            x_title = "Mese"
        elif group_by == "week":
            daily_stats["period"] = daily_stats["date"].dt.to_period("W")  # Raggruppa per settimana
            x_title = "Settimana"
        else:
            raise ValueError("Valore non valido per 'group_by'. Usa 'month' o 'week'")

        # Calcola la media e la deviazione standard per ogni periodo
        stats = daily_stats.groupby("period")["value"].agg(["mean"]).reset_index()
        if(measure == "total_sleep_time"):
            stats['mean']= stats['mean'].dt.total_seconds() / 60
        stats["period"] = stats["period"].astype(str)  # Converti il periodo in stringa per il grafico

        # Crea il grafico
        fig = go.Figure()

        # Barra della media con errore (deviazione standard)
        fig.add_trace(go.Bar(
            x=stats["period"],
            y=stats["mean"],
            name="Media",
            marker_color="orange"
        ))

        if(measure == 'total_sleep_time'):
            # Personalizza il layout
            fig.update_layout(
                title=f"Media di {measure} per {group_by.capitalize()}",
                xaxis_title=x_title,
                yaxis_title=f'{measure} (Minutes)',
                template="plotly_white",
                height=600,
                width=1000,
                showlegend=True
            )
        else:
            # Personalizza il layout
            fig.update_layout(
                title=f"Media di {measure} per {group_by.capitalize()}",
                xaxis_title=x_title,
                yaxis_title=f'{measure}',
                template="plotly_white",
                height=600,
                width=1000,
                showlegend=True
            )

        # Salva in HTML o PNG, se richiesto
        if generate_html:
            file_name = f"AggregatedData_{measure}_{start_date}_{end_date}"
            pio.write_html(fig, file='graphs/'+file_name+'.html', auto_open=True)

        if save_png:
            file_name = f"AggregatedData_{measure}_{group_by}_{start_date}_{end_date}.png"
            full_path = f"../graphs/{file_name}"
            fig.write_image(full_path, format="png", width=1000, height=600)
            print(f"Grafico salvato come PNG in: {full_path}")

        # Mostra il grafico
        fig.show()

    def plot_States(self, start_date, end_date, group_by="month", generate_html=False, save_png= False):
        
        # Genera una lista di date tra start_date ed end_date
        date_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')
        dates = []
        light_values = []
        rem_values = []
        deep_values = []

        for specific_date in date_list:
            # Ottieni la misura per una specifica notte
            sleepsMeasure = self.getMeasuresOfSpecificNight(specific_date)
            if len(sleepsMeasure) > 0:
                # Estrai i valori dalla lista di misure
                specific_light_values = sleepsMeasure['light_sleep_duration'].tolist()
                specific_rem_values = sleepsMeasure['rem_sleep_duration'].tolist()
                specific_deep_values = sleepsMeasure['deep_sleep_duration'].tolist()
                light_values.extend(specific_light_values)  # Aggiungi tutti i valori (può essere una lista)
                rem_values.extend(specific_rem_values)  # Aggiungi tutti i valori (può essere una lista)
                deep_values.extend(specific_deep_values)  # Aggiungi tutti i valori (può essere una lista)
                dates.extend([specific_date] * len(specific_light_values))  # Ripeti la data per ogni valore

        # Controlla se ci sono dati sufficienti
        if len(dates) == 0 or len(light_values) == 0 or len(rem_values) == 0 or len(deep_values) == 0:
            print("Nessun dato disponibile per il periodo selezionato.")
            return

        # Crea un DataFrame per l'elaborazione
        df = pd.DataFrame({"dates": pd.to_datetime(dates), "light_sleep_duration": pd.to_timedelta(light_values, unit="s"), "rem_sleep_duration":  pd.to_timedelta(rem_values, unit="s") , "deep_sleep_duration":  pd.to_timedelta(deep_values, unit="s")})

        # Raggruppa per mese o settimana
        if group_by == "month":
            df["period"] = df["dates"].dt.to_period("M")
        elif group_by == "week":
            df["period"] = df["dates"].dt.to_period("W")
        elif group_by == "day":
            df["period"] = df["dates"].dt.to_period("D")
        else:
            raise ValueError("group_by deve essere 'month' o 'week'")
        
        # Calcola la somma delle durate per ciascun periodo
        grouped_data = df.groupby("period")[["light_sleep_duration", "rem_sleep_duration", "deep_sleep_duration"]].sum().reset_index()
        
        # Estrai i dati per il plot
        periods = grouped_data["period"].astype(str)
        light_sleep = grouped_data["light_sleep_duration"].dt.total_seconds() / 60
        rem_sleep = grouped_data["rem_sleep_duration"].dt.total_seconds() / 60
        deep_sleep = grouped_data["deep_sleep_duration"].dt.total_seconds() / 60
        
        # Crea il grafico a barre stacked
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=periods,
            y=light_sleep,
            name="Light Sleep",
            marker_color="skyblue"
        ))
        
        fig.add_trace(go.Bar(
            x=periods,
            y=rem_sleep,
            name="REM Sleep",
            marker_color="orange"
        ))
        
        fig.add_trace(go.Bar(
            x=periods,
            y=deep_sleep,
            name="Deep Sleep",
            marker_color="purple"
        ))
        
        # Personalizza il layout
        fig.update_layout(
            title=f"Sleep Durations by {group_by.capitalize()}",
            xaxis_title="Period",
            yaxis_title="Duration (minutes)",
            barmode="stack",  # Modalità stacked
            template="plotly_white",
            height=600,
            width=1000
        )
        
        if generate_html:
            file_name = f"States_{start_date}_{end_date}"
            pio.write_html(fig, file='graphs/'+file_name+'.html', auto_open=True)
        
        if save_png:
            file_name = f"SleepStates_{group_by}_{start_date}_{end_date}.png"
            full_path = f"../graphs/{file_name}"
            fig.write_image(full_path, format="png", width=1000, height=600)
            print(f"Grafico salvato come PNG in: {full_path}")
        fig.show()

    def plot_States_Percentages(self, start_date, end_date, group_by="month", generate_html=False, save_png= False):
        
        # Genera una lista di date tra start_date ed end_date
        date_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')
        dates = []
        light_values = []
        rem_values = []
        deep_values = []

        for specific_date in date_list:
            # Ottieni la misura per una specifica notte
            sleepsMeasure = self.getMeasuresOfSpecificNight(specific_date)
            if len(sleepsMeasure) > 0:
                # Estrai i valori dalla lista di misure
                specific_light_values = sleepsMeasure['light_sleep_duration'].tolist()
                specific_rem_values = sleepsMeasure['rem_sleep_duration'].tolist()
                specific_deep_values = sleepsMeasure['deep_sleep_duration'].tolist()
                light_values.extend(specific_light_values)  # Aggiungi tutti i valori
                rem_values.extend(specific_rem_values)  # Aggiungi tutti i valori
                deep_values.extend(specific_deep_values)  # Aggiungi tutti i valori
                dates.extend([specific_date] * len(specific_light_values))  # Ripeti la data per ogni valore

        # Controlla se ci sono dati sufficienti
        if len(dates) == 0 or len(light_values) == 0 or len(rem_values) == 0 or len(deep_values) == 0:
            print("Nessun dato disponibile per il periodo selezionato.")
            return

        # Crea un DataFrame per l'elaborazione
        df = pd.DataFrame({
            "dates": pd.to_datetime(dates),
            "light_sleep_duration": light_values,
            "rem_sleep_duration": rem_values,
            "deep_sleep_duration": deep_values
        })

        # Raggruppa per mese o settimana
        if group_by == "month":
            df["period"] = df["dates"].dt.to_period("M")
        elif group_by == "week":
            df["period"] = df["dates"].dt.to_period("W")
        elif group_by == "day":
            df["period"] = df["dates"].dt.to_period("D")
        else:
            raise ValueError("group_by deve essere 'month' o 'week' o 'day'")
        
        # Calcola la somma delle durate per ciascun periodo
        grouped_data = df.groupby("period")[["light_sleep_duration", "rem_sleep_duration", "deep_sleep_duration"]].sum().reset_index()

        # Calcola le percentuali
        grouped_data["total_duration"] = grouped_data[["light_sleep_duration", "rem_sleep_duration", "deep_sleep_duration"]].sum(axis=1)
        grouped_data["LightSleepPercentage"] = (grouped_data["light_sleep_duration"] / grouped_data["total_duration"]) * 100
        grouped_data["RemSleepPercentage"] = (grouped_data["rem_sleep_duration"] / grouped_data["total_duration"]) * 100
        grouped_data["DeepSleepPercentage"] = (grouped_data["deep_sleep_duration"] / grouped_data["total_duration"]) * 100

        # Estrai i dati per il plot
        periods = grouped_data["period"].astype(str)
        light_sleep = grouped_data["LightSleepPercentage"]
        rem_sleep = grouped_data["RemSleepPercentage"]
        deep_sleep = grouped_data["DeepSleepPercentage"]

        # Crea il grafico a barre stacked
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=periods,
            y=light_sleep,
            name="Light Sleep (%)",
            marker_color="skyblue"
        ))
        
        fig.add_trace(go.Bar(
            x=periods,
            y=rem_sleep,
            name="REM Sleep (%)",
            marker_color="orange"
        ))
        
        fig.add_trace(go.Bar(
            x=periods,
            y=deep_sleep,
            name="Deep Sleep (%)",
            marker_color="purple"
        ))

        # Personalizza il layout
        fig.update_layout(
            title=f"Sleep Percentages by {group_by.capitalize()}",
            xaxis_title="Period",
            yaxis_title="Percentage (%)",
            barmode="stack",  # Modalità stacked
            template="plotly_white",
            height=600,
            width=1000
        )
        
        if generate_html:
            file_name = f"Sleep_Percentages_{start_date}_{end_date}"
            pio.write_html(fig, file='graphs/' + file_name + '.html', auto_open=True)

        if save_png:
            file_name = f"Sleep_Percentages_{group_by}_{start_date}_{end_date}.png"
            full_path = f"../graphs/{file_name}"
            fig.write_image(full_path, format="png", width=1000, height=600)
            print(f"Grafico salvato come PNG in: {full_path}")
        fig.show()
    
def mergeAggregatedData(aggregatedGranular, aggregatedWithings):
    merged_aggregated = {}
    
    granularDict = aggregatedGranular.getMeasureDict()
    withingsDict = aggregatedWithings.getMeasureDict()

    # Otteniamo l'unione di tutte le date presenti in entrambi i dizionari
    all_dates = set(granularDict.keys()).union(set(withingsDict.keys()))

    for date in all_dates:
        if date in granularDict and date in withingsDict:
            granularDF = aggregatedGranular.getMeasuresOfSpecificNight(date)
            withingsDf = aggregatedWithings.getMeasuresOfSpecificNight(date)

            # Merge incrociato tra df1 e df2
            overlap = pd.merge(granularDF, withingsDf, how='cross')

            # Verifica se c'è overlap
            overlap['is_overlap'] = (overlap['date_start_x'] < overlap['date_end_y']) & (overlap['date_end_x'] > overlap['date_start_y'])

            # Righe con overlap
            overlapping_rows = overlap[overlap['is_overlap']]

            # Selezioniamo colonne specifiche per il risultato combinato
            columns_from_df1 = ['manual_sleep_duration','undefined_sleep_duration','hr_average_x','hr_max_x','hr_min_x','rmssd_average','rmssd_max','rmssd_min',
            'rr_average_x','rr_max_x','rr_min_x','sdnn_1_average','sdnn_1_max','sdnn_1_min','snoring_average','snoring_max','snoring_min',
            'mvt_score_average_x','mvt_score_max','mvt_score_min','wakeup_count_x','out_of_bed_count_x','out_of_bed_time','out_of_bed']
            columns_from_df2 = ['date_y','date_start_y','date_end_y','nap_y','light_sleep_duration_y','deep_sleep_duration_y','rem_sleep_duration_y','wakeup_duration_y','total_sleep_time_y',
            'nb_rem_episodes','sleep_efficiency','sleep_latency','total_timeinbed','apnea_hypopnea_index','breathing_disturbances_intensity','asleepduration',
            'mvt_active_duration','night_events','snoring','wakeup_latency','waso','snoringepisodecount','sleep_score','model_y','model_id_y']

            

            # Uniamo le righe con overlap
            overlap_combined = overlapping_rows[columns_from_df1 + columns_from_df2]

            overlap_combined = overlap_combined.rename(columns={
                                    'hr_average_x': 'hr_average','hr_max_x': 'hr_max',
                                    'hr_min_x': 'hr_min','rr_average_x': 'rr_average',
                                    'rr_max_x': 'rr_max','out_of_bed_count_x': 'out_of_bed_count',
                                    'date_y': 'date','date_start_y': 'date_start',
                                    'date_end_y': 'date_end', 'nap_y': 'nap',
                                    'total_sleep_time_y': 'total_sleep_time', 'model_y': 'model',
                                    'model_id_y': 'model_id', 'rr_min_x': 'rr_min',
                                    'light_sleep_duration_y' : 'light_sleep_duration', 'deep_sleep_duration_y' : 'deep_sleep_duration',
                                    'rem_sleep_duration_y' : 'rem_sleep_duration', 'wakeup_duration_y' : 'wakeup_duration',
                                    'mvt_score_average_x' : 'mvt_score_average', 'wakeup_count_x' : 'wakeup_count'})

            # DataFrame per le righe non sovrapposte di df1
            no_overlap_df1 = granularDF[~granularDF.index.isin(overlap_combined.index)]
            # DataFrame per le righe non sovrapposte di df2
            no_overlap_df2 = withingsDf[~withingsDf.index.isin(overlap_combined.index)]

            # Creiamo il DataFrame finale che include anche le righe non sovrapposte di df2
            final_df = pd.concat([overlap_combined, no_overlap_df1, no_overlap_df2])

            # Lista delle colonne da escludere
            colonne_davanti = ['date','date_start','date_end', 'nap']
            # Ottieni i nomi delle colonne rimanenti
            colonne_rimanenti = [col for col in final_df.columns if col not in colonne_davanti]
            final_df = final_df[colonne_davanti + colonne_rimanenti]
            df_filtered = final_df.dropna(axis=1, how='all')
            merged_aggregated[date] = df_filtered

        elif date in granularDict:
            df = aggregatedGranular.getMeasuresOfSpecificNight(date)

            #nuovi campi
            colonneAggiuntive=['nb_rem_episodes','sleep_efficiency','sleep_latency','total_timeinbed','apnea_hypopnea_index',
                       'breathing_disturbances_intensity','asleepduration','mvt_active_duration','night_events','snoring','wakeup_latency','waso','snoringepisodecount','sleep_score']

            # Lista delle colonne da escludere
            colonne_davanti = ['date','date_start','date_end', 'nap']
            # Ottieni i nomi delle colonne rimanenti
            colonne_rimanenti = [col for col in df.columns if col not in colonne_davanti]
            df = df[colonne_davanti + colonne_rimanenti]
            df[colonneAggiuntive] = np.nan
            df_filtered = df.dropna(axis=1, how='all')
            merged_aggregated[date] = df_filtered

        elif date in withingsDict:
            
            #non ho trovato una corrispondenza nei granulari quindi riempo il df solo con i dati di Withings
            df = aggregatedWithings.getMeasuresOfSpecificNight(date)
            colonneAggiuntive=['rmssd_average', 'rmssd_max','rmssd_min','sdnn_1_average','sdnn_1_max','sdnn_1_min','snoring_average','snoring_max','snoring_min','out_of_bed_time','out_of_bed']

            df['manual_sleep_duration'] = 0
            df['undefined_sleep_duration'] = 0
            df['mvt_score_max'] = df['mvt_score_average']
            df['mvt_score_min'] = df['mvt_score_average']

            # Lista delle colonne da escludere
            colonne_davanti = ['date','date_start','date_end', 'nap']
            # Ottieni i nomi delle colonne rimanenti
            colonne_rimanenti = [col for col in df.columns if col not in colonne_davanti]
            df = df[colonne_davanti + colonne_rimanenti]
            df[colonneAggiuntive] = np.nan
            df_filtered = df.dropna(axis=1, how='all')
            merged_aggregated[date] = df_filtered

    # Step 1: Combina tutti i DataFrame in un unico DataFrame
    combined_df = pd.concat(merged_aggregated.values(), ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values(by='date_end', ignore_index=True)

    # Specifica le colonne in cui vuoi riempire i valori nulli
    # Lista delle colonne da escludere
    columnsNotToCheck = ['date_start','date_end', 'nap', 'out_of_bed', 'model', 'model_id', 'night_events','date', 'sleep_score']
    # Ottieni i nomi delle colonne rimanenti
    columns_to_check = [col for col in combined_df.columns if col not in columnsNotToCheck]

    # Funzione per riempire i valori nulli e tracciare le righe modificate
    def fill_nulls_and_get_modified_rows(df, columns_to_check):
        modified_rows_df = []
        zeros_rows_df = []

        # Iteriamo sulle colonne da controllare
        for col in columns_to_check:
            if col == 'out_of_bed_time':
                # Gestisci 'out_of_bed_time' separatamente, poiché si riempie con 0
                mask_null = df[col].isnull()
                df.loc[mask_null, col] = timedelta(0)
                modified_rows_df.append(df[mask_null])  # Aggiungi alle righe modificate

            # Per le altre colonne
            elif col in ['hr_average', 'hr_max', 'hr_min', 'rmssd_average', 'rmssd_max', 'rmssd_min',
                        'rr_average', 'rr_max', 'rr_min', 'sdnn_1_average', 'sdnn_1_max', 'sdnn_1_min',
                        'snoring_average', 'snoring_max', 'snoring_min', 'mvt_score_average', 'mvt_score_max',
                        'mvt_score_min']:
                # Se ci sono valori nulli, riempiamo con la media delle ultime 7 righe per ogni valore di 'nap'
                mask_null = df[col].isnull()
                if mask_null.any():  # Se ci sono valori nulli
                    for nap_value in df['nap'].unique():
                        # Trova le righe per quel valore di 'nap'
                        rows_before_nap = df[df['nap'] == nap_value].tail(7)
                        
                        # Calcola la media delle ultime 7 righe
                        mean_value = rows_before_nap[col].mean()
                        
                        # Assegna la media ai valori nulli per quella colonna e 'nap'
                        df.loc[mask_null & (df['nap'] == nap_value), col] = mean_value
                    
                    modified_rows_df.append(df[mask_null])  # Aggiungi alle righe modificate
            else:
                # Per altre colonne, riempiamo i valori nulli con la media delle ultime 7 righe
                mask_null = df[col].isnull()
                if mask_null.any():  # Se ci sono valori nulli
                    for nap_value in df['nap'].unique():
                        # Trova le righe per quel valore di 'nap'
                        rows_before_nap = df[df['nap'] == nap_value].tail(7)
                        
                        # Calcola la media delle ultime 7 righe
                        mean_value = rows_before_nap[col].mean()
                        
                        # Assegna la media ai valori nulli per quella colonna e 'nap'
                        df.loc[mask_null & (df['nap'] == nap_value), col] = mean_value
                    
                    modified_rows_df.append(df[mask_null])  # Aggiungi alle righe modificate

        # Identifica righe con zero
        zeros_mask = df[columns_to_check].eq(0).any(axis=1)
        zeros_rows_df.append(df[zeros_mask])  # Aggiungi alle righe con zero

        return df, pd.concat(modified_rows_df), pd.concat(zeros_rows_df)

    # Applica la funzione al DataFrame combinato
    combined_df, modified_rows_df, zeros_rows_df = fill_nulls_and_get_modified_rows(combined_df, columns_to_check)


    # Crea un dizionario che raggruppa le righe per data
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')
    grouped_dict = {date: group for date, group in combined_df.groupby('date')}
    return grouped_dict, modified_rows_df, zeros_rows_df
