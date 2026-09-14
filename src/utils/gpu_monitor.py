"""Read-only NVIDIA telemetry. Never sets power, clocks or fan speed."""
from pathlib import Path
import csv
import json
import subprocess
import threading
import time


def snapshot():
    fields=['index','uuid','name','power.limit','power.max_limit','power.draw','temperature.gpu','fan.speed','utilization.gpu','memory.used']
    output=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[]
    for values in csv.reader(output.splitlines()):
        row={key:value.strip() for key,value in zip(fields,values)}
        row['index']=int(row['index'])
        if row['index'] in range(4):rows.append(row)
    if len(rows)!=4:raise RuntimeError('Four GPU telemetry rows required')
    return rows


class ThermalPolicy:
    def __init__(self,warning_temperature_c=78,emergency_stop_temperature_c=89,required_consecutive_samples=1,telemetry_interval_seconds=10):
        self.warning_temperature_c=warning_temperature_c;self.emergency_stop_temperature_c=emergency_stop_temperature_c
        self.required_consecutive_samples=required_consecutive_samples;self.telemetry_interval_seconds=telemetry_interval_seconds

class ThermalCounter:
    def __init__(self,policy=None):self.policy=policy or ThermalPolicy();self.counts={i:0 for i in range(4)}
    def update(self,rows):
        warning=[];emergency=[]
        for row in rows:
            i=row['index'];temp=float(row['temperature.gpu'])
            if temp>=self.policy.warning_temperature_c: warning.append(i)
            self.counts[i]=self.counts[i]+1 if temp>=self.policy.emergency_stop_temperature_c else 0
            if self.counts[i]>=self.policy.required_consecutive_samples: emergency.append(i)
        return {'warning_gpus':warning,'emergency_gpus':emergency,'emergency_stop':bool(emergency)}


class GPUMonitor:
    def __init__(self,directory,telemetry_name='gpu_telemetry.jsonl',interval_seconds=10,policy=None,stop_enabled=True):
        self.directory=Path(directory);self.telemetry_name=telemetry_name;self.done=threading.Event();self.thermal_stop=threading.Event()
        self.interval_seconds=float(interval_seconds)
        if self.interval_seconds<=0: raise ValueError('interval_seconds must be positive')
        self.policy=policy or ThermalPolicy(telemetry_interval_seconds=interval_seconds);self.stop_enabled=bool(stop_enabled);self.error=None;self.counter=ThermalCounter(self.policy);self.thread=None;self.emergency_stop=False
    def measure(self):
        rows=snapshot()
        for row in rows:
            p=self.directory/f'memory_rank{row["index"]}.json'
            row['allocator']=json.loads(p.read_text()) if p.exists() else None
        decision=self.counter.update(rows);stop=decision['emergency_stop'];self.emergency_stop=stop
        with (self.directory/self.telemetry_name).open('a') as f:
            f.write(json.dumps(dict(timestamp=time.time(),gpus=rows,thermal_counts=self.counter.counts,thermal_warning=bool(decision['warning_gpus']),warning_gpus=decision['warning_gpus'],emergency_gpus=decision['emergency_gpus'],thermal_stop=stop if self.stop_enabled else False,thermal_stop_enabled=self.stop_enabled,thermal_policy=self.policy.__dict__))+'\n')
        if stop and self.stop_enabled:self.thermal_stop.set()
    def start(self):
        self.measure()
        def loop():
            while not self.done.wait(self.interval_seconds):
                try:self.measure()
                except Exception as exc:self.error=repr(exc);return
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def close(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=20)
