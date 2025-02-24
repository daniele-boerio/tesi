import pandas as pd

measures = ['date','date_start','date_end','nap','nb_rem_episodes','sleep_efficiency',
            'sleep_latency','total_sleep_time','total_timeinbed','apnea_hypopnea_index',
            'breathing_disturbances_intensity','hr_average','hr_max','hr_min',
            'light_sleep_duration','deep_sleep_duration','asleepduration','mvt_active_duration',
            'mvt_score_average','night_events','snoring','rem_sleep_duration','sleep_score',
            'out_of_bed_count','wakeup_count','wakeup_duration','wakeup_latency','waso',
            'rr_average','rr_max','rr_min','snoringepisodecount','withings_index','model','model_id']


class AggregatedDataFromWithings:
    #This class accepts a dictionary of nights, where a night is a dictionary of measurements from the previous night to the morning of a specific day of the year.
    def __init__(self, aggWithings):
        aggWithings['start_sleep'] = pd.to_datetime(aggWithings['start_sleep'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Rome')
        aggWithings['end_sleep'] = pd.to_datetime(aggWithings['end_sleep'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Rome')
        # Verifica se le colonne 'startValue' e 'endValue' esistono nel DataFrame
        if 'start_sleep' in aggWithings.columns and 'end_sleep' in aggWithings.columns:
            # Converti 'startValue' e 'endValue' in datetime (se non lo sono già)
            aggWithings['start_sleep'] = pd.to_datetime(aggWithings['start_sleep'], errors='coerce')
            aggWithings['end_sleep'] = pd.to_datetime(aggWithings['end_sleep'], errors='coerce')

            # Crea una colonna per il controllo del "nap"
            aggWithings['nap'] = False

            # Calcola il giorno e l'ora per 'startValue' e 'endValue'
            start_day = aggWithings['start_sleep'].dt.day
            end_day = aggWithings['end_sleep'].dt.day
            start_hour = aggWithings['start_sleep'].dt.hour
            end_hour = aggWithings['end_sleep'].dt.hour

            # Verifica la condizione "nap" per tutte le righe
            nap_condition = (start_day == end_day) & (start_hour > 10) & (end_hour < 23)
            
            # Imposta la colonna 'nap' in base alla condizione
            aggWithings.loc[nap_condition, 'nap'] = True
        aggWithings = aggWithings.rename(columns={'Time':'date','start_sleep': 'date_start', 'end_sleep':'date_end',
                                                  'lightsleepduration':'light_sleep_duration','deepsleepduration': 'deep_sleep_duration',
                                                    'mvt_score_avg':'mvt_score_average', 'remsleepduration':'rem_sleep_duration',
                                                    'wakeupcount':'wakeup_count', 'wakeupduration':'wakeup_duration'})
        aggWithings = aggWithings.drop(columns=['subject', 'withings_index'])
        self.aggWithings = aggWithings
    
    #return the copy of the dictionary with aggregated data of each day of the year
    def getMeasureDict(self):
        # Gruppo per il campo 'Time' e creo il dizionario
        grouped = self.aggWithings.copy().groupby('date')
        time_dict = {key: group.reset_index(drop=True) for key, group in grouped}
    
        return time_dict
    
    #return the dataframe with aggregated data of each day of the year
    def getMeasureDf(self):
        return self.aggWithings.copy()
    
    def getMeasuresOfSpecificNight(self, specific_date):
        # Extract values on a specific data
        measureDict = self.getMeasureDict() 
        if specific_date in measureDict:
            df = pd.DataFrame(measureDict[specific_date])
            return df
        else:
            required_columns = ['date','date_start','date_end','nap','nb_rem_episodes','sleep_efficiency',
                                'sleep_latency','total_sleep_time','total_timeinbed','apnea_hypopnea_index',
                                'breathing_disturbances_intensity','hr_average','hr_max','hr_min',
                                'light_sleep_duration','deep_sleep_duration','asleepduration','mvt_active_duration',
                                'mvt_score_average','night_events','snoring','rem_sleep_duration','sleep_score',
                                'out_of_bed_count','wakeup_count','wakeup_duration','wakeup_latency','waso',
                                'rr_average','rr_max','rr_min','snoringepisodecount','withings_index','model','model_id']
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
                required_columns = ['date','date_start','date_end','nap','nb_rem_episodes','sleep_efficiency',
                                'sleep_latency','total_sleep_time','total_timeinbed','apnea_hypopnea_index',
                                'breathing_disturbances_intensity','hr_average','hr_max','hr_min',
                                'light_sleep_duration','deep_sleep_duration','asleepduration','mvt_active_duration',
                                'mvt_score_average','night_events','snoring','rem_sleep_duration','sleep_score',
                                'out_of_bed_count','wakeup_count','wakeup_duration','wakeup_latency','waso',
                                'rr_average','rr_max','rr_min','snoringepisodecount','withings_index','model','model_id']
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