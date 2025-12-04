import pandas as pd
import numpy as np
import gnss_lib_py as glp
from datetime import datetime, timedelta


class CSVToNavData:
    
    GNSS_ID_MAP = {
        0: 'gps',
        1: 'sbas',
        2: 'galileo',
        3: 'beidou',
        5: 'qzss',
        6: 'glonass'
    }
    
    GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.load_csv()
    
    def load_csv(self):
        print(f"Loading CSV from {self.csv_path}...")
        
        dtypes = {
            'gnssId': 'int8',
            'svId': 'int16',
            'sigId': 'int8',
            'cn0': 'float32',
            'elev': 'float32',
            'azim': 'float32',
            'pseudorange': 'float32',
            'doppler': 'float32',
            'prRes': 'float32',
            'clkB': 'float32',
            'clkD': 'float32',
            'lon': 'float32',
            'lat': 'float32',
            'height': 'float32',
            'ecefX': 'float32',
            'ecefY': 'float32',
            'ecefZ': 'float32',
        }
        self.df = pd.read_csv(
            self.csv_path,
            parse_dates=['time'],
            date_format="%Y-%m-%d %H:%M:%S",
            dtype=dtypes,
        )
        print(f"Loaded {len(self.df)} rows")
        
        return self.df
    
    @classmethod
    def datetime_to_gps_millis(cls, dt):
        if isinstance(dt, pd.Timestamp) or isinstance(dt, datetime):
            delta = dt - cls.GPS_EPOCH
        else:
            dt_obj = pd.to_datetime(dt)
            delta = dt_obj - cls.GPS_EPOCH
        gps_millis = delta.total_seconds() * 1000.0
        return gps_millis
    
    @classmethod
    def gps_millis_to_datetime(cls, gps_millis):
        if isinstance(gps_millis, (list, np.ndarray)):
            return [cls.GPS_EPOCH + timedelta(milliseconds=float(gm)) for gm in gps_millis]
        else:
            return cls.GPS_EPOCH + timedelta(milliseconds=float(gps_millis))
    
    def convert_to_navdata(self, add_receiver_states=True):
        print("\nConverting to NavData format...")
        
        navdata = glp.NavData()
        
        print("  Converting timestamps...")
        if pd.api.types.is_datetime64_any_dtype(self.df['time']):
            gps_millis = ((self.df['time'] - self.GPS_EPOCH).dt.total_seconds() * 1000.0).values
        else:
            gps_millis = self.df['time'].apply(self.datetime_to_gps_millis).values
        
        navdata['gps_millis'] = gps_millis
        
        print("  Adding measurements...")
        gnss_id_str = [self.GNSS_ID_MAP.get(gid, f'unknown_{gid}') 
                       for gid in self.df['gnssId'].values]
        navdata['gnss_id'] = np.array(gnss_id_str)
        
        navdata['sv_id'] = self.df['svId'].values.astype(np.int64)
        navdata['sig_id'] = self.df['sigId'].values.astype(np.int64)
        
        navdata['raw_pr_m'] = self.df['pseudorange'].values
        navdata['doppler_hz'] = self.df['doppler'].values
        navdata['cn0_dbhz'] = self.df['cn0'].values
        
        navdata['el_sv_deg'] = self.df['elev'].values
        navdata['az_sv_deg'] = self.df['azim'].values
        
        navdata['pr_residual_m'] = self.df['prRes'].values
        
        navdata['b_rx_ns'] = self.df['clkB'].values
        navdata['b_rx_m'] = self.df['clkB'].values * 1e-9 * 299792458.0
        
        navdata['clock_drift_ns_s'] = self.df['clkD'].values
        
        if add_receiver_states:
            
            if all(col in self.df.columns for col in ['ecefX', 'ecefY', 'ecefZ']):
                navdata['x_rx_m'] = self.df['ecefX'].values / 100.0
                navdata['y_rx_m'] = self.df['ecefY'].values / 100.0
                navdata['z_rx_m'] = self.df['ecefZ'].values / 100.0
            
            navdata['lat_rx_deg'] = self.df['lat'].values
            navdata['lon_rx_deg'] = self.df['lon'].values
            navdata['alt_rx_m'] = self.df['height'].values / 100.0  # cm to m
        
        navdata['timestamp_str'] = self.df['time'].values
        
        print(f"\nCreated NavData with {navdata.shape[1]} measurements")
        print(f"  Unique epochs: {len(np.unique(gps_millis))}")
        print(f"  Constellations: {set(gnss_id_str)}")
        print(f"  Satellites: {len(np.unique(self.df['svId']))}")
        
        return navdata
    
    def get_state_estimates(self):
        grouped = self.df.groupby('time').first().reset_index()
        
        state_estimates = glp.NavData()
        
        if pd.api.types.is_datetime64_any_dtype(grouped['time']):
            gps_millis = ((grouped['time'] - self.GPS_EPOCH).dt.total_seconds() * 1000.0).values
        else:
            gps_millis = grouped['time'].apply(self.datetime_to_gps_millis).values
        
        state_estimates['gps_millis'] = gps_millis
        
        if all(col in grouped.columns for col in ['ecefX', 'ecefY', 'ecefZ']):
            state_estimates['x_rx_m'] = grouped['ecefX'].values / 100.0
            state_estimates['y_rx_m'] = grouped['ecefY'].values / 100.0
            state_estimates['z_rx_m'] = grouped['ecefZ'].values / 100.0
        
        state_estimates['b_rx_m'] = grouped['clkB'].values * 1e-9 * 299792458.0
        
        state_estimates['lat_rx_deg'] = grouped['lat'].values
        state_estimates['lon_rx_deg'] = grouped['lon'].values
        state_estimates['alt_rx_m'] = grouped['height'].values / 100.0
        
        print(f"\nExtracted {state_estimates.shape[1]} state estimates")
        
        return state_estimates