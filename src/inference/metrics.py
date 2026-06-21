import threading
import time
from pathlib import Path
import psutil
import csv


class Metrics:
    total_inference_time = 0 # Tiempo de inferencia (s)
    inference_latencies = [] # Latencia por inferencia (s)
    cpu_usage = [] # Uso de CPU (%)
    ram_usage = [] # Uso de RAM (MiB)
    disk_usage = [] # Uso de disco (actividad en %)
    
    monitor_thread = None
    monitor_stop_event = threading.Event()
    monitor_lock = threading.Lock()
    last_disk_io = None
    last_disk_sample_time = None
    
    # Inferencias por segundo
    @classmethod
    def inferences_per_second(cls) -> float:
        if cls.total_inference_time == 0:
            return 0.0
        else:
            return len(cls.inference_latencies) / cls.total_inference_time
    

    
    @classmethod
    def add_inference_time(cls, time: float):
        cls.total_inference_time += time
        cls.inference_latencies.append(time)

    # Promedio de latencias
    @classmethod
    def average_inference_time(cls) -> float:
        if len(cls.inference_latencies) == 0:
            return 0.0
        else:
            return cls.total_inference_time / len(cls.inference_latencies)
    
    # Latencia máxima
    @classmethod
    def max_inference_time(cls) -> float:
        if len(cls.inference_latencies) == 0:
            return 0.0
        else:
            return max(cls.inference_latencies)
    
    
    
    @classmethod
    def add_cpu_usage(cls, usage: float):
        cls.cpu_usage.append(usage)
    
    # Promedio de uso de CPU
    @classmethod
    def average_cpu_usage(cls) -> float:
        # La primera muestra siempre es 0.0, por lo que no se considera en el promedio
        if len(cls.cpu_usage) <= 1:
            return 0.0
        else:
            return sum(cls.cpu_usage[1:]) / (len(cls.cpu_usage) - 1)

    # Uso máximo de CPU
    @classmethod
    def max_cpu_usage(cls) -> float:
        if len(cls.cpu_usage) == 0:
            return 0.0
        else:
            return max(cls.cpu_usage)
    
    
    
    @classmethod
    def add_ram_usage(cls, usage: float):
        cls.ram_usage.append(usage)
    
    # Promedio de uso de RAM
    @classmethod
    def average_ram_usage(cls) -> float:
        if len(cls.ram_usage) == 0:
            return 0.0
        else:
            return sum(cls.ram_usage) / len(cls.ram_usage)
    
    # Uso máximo de RAM
    @classmethod
    def max_ram_usage(cls) -> float:
        if len(cls.ram_usage) == 0:
            return 0.0
        else:
            return max(cls.ram_usage)
    
    
    
    @classmethod
    def add_disk_usage(cls, usage: float):
        cls.disk_usage.append(usage)

    # Promedio de uso de disco
    @classmethod
    def average_disk_usage(cls) -> float:
        # La primera muestra siempre es -1.0, por lo que no se considera en el promedio
        if len(cls.disk_usage) <= 1:
            return 0.0
        else:
            return sum(cls.disk_usage[1:]) / (len(cls.disk_usage) - 1)

    # Uso máximo de disco
    @classmethod
    def max_disk_usage(cls) -> float:
        if len(cls.disk_usage) == 0:
            return 0.0
        else:
            return max(cls.disk_usage)



    @classmethod
    def print_metrics(cls):
        print("=== Métricas de inferencia ===")
        print(f"Número de inferencias: {len(cls.inference_latencies)}")
        print(f"Tiempo de inferencia: {cls.total_inference_time:.4f} segundos")
        print(f"Promedio de latencias: {cls.average_inference_time():.4f} segundos")
        print(f"Latencia máxima: {cls.max_inference_time():.4f} segundos")
        print(f"Inferencias por segundo: {cls.inferences_per_second():.4f} inf/s")
        print(f"Promedio de uso de CPU: {cls.average_cpu_usage():.2f}%")
        print(f"Uso máximo de CPU: {cls.max_cpu_usage():.2f}%")
        print(f"Promedio de uso de RAM: {cls.average_ram_usage():.2f} MiB")
        print(f"Uso máximo de RAM: {cls.max_ram_usage():.2f} MiB")
        print(f"Promedio de uso de disco: {cls.average_disk_usage():.2f}%")
        print(f"Uso máximo de disco: {cls.max_disk_usage():.2f}%")
        print("=============================")
    
    @classmethod
    def sample_system_metrics(cls):
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_used_mib = psutil.virtual_memory().used / (1024 * 1024)
        
        # Cálculo del porcentaje de uso del disco basado en el tiempo ocupado del disco
        current_disk_io = psutil.disk_io_counters(perdisk=True)
        current_time = time.monotonic()
        disk_percent = -1.0

        if current_disk_io is not None and cls.last_disk_io is not None and cls.last_disk_sample_time is not None:
            elapsed_seconds = current_time - cls.last_disk_sample_time
            if elapsed_seconds > 0:
                disk_percentages = []

                for disk_name, current_disk_stats in current_disk_io.items():
                    previous_disk_stats = cls.last_disk_io.get(disk_name)
                    if previous_disk_stats is None:
                        continue

                    # Calcular el tiempo ocupado del disco en milisegundos
                    if hasattr(current_disk_stats, "busy_time") and hasattr(previous_disk_stats, "busy_time"):
                        current_disk_time = current_disk_stats.busy_time
                        previous_disk_time = previous_disk_stats.busy_time
                    else:
                        current_disk_time = current_disk_stats.read_time + current_disk_stats.write_time
                        previous_disk_time = previous_disk_stats.read_time + previous_disk_stats.write_time

                    disk_time_delta_ms = float(current_disk_time - previous_disk_time)
                    if disk_time_delta_ms >= 0:
                        disk_percentages.append((disk_time_delta_ms / (elapsed_seconds * 1000.0)) * 100.0)

                if len(disk_percentages) > 0:
                    disk_percent = min(max(disk_percentages), 100.0)

        cls.last_disk_io = current_disk_io
        cls.last_disk_sample_time = current_time
        # Fin cálculo del porcentaje de uso del disco

        with cls.monitor_lock:
            cls.add_cpu_usage(float(cpu_percent))
            cls.add_ram_usage(float(ram_used_mib))
            cls.add_disk_usage(float(disk_percent))

    @classmethod
    def _monitor_loop(cls, interval_seconds: float):
        while not cls.monitor_stop_event.is_set():
            cls.sample_system_metrics()
            if cls.monitor_stop_event.wait(timeout=interval_seconds):
                break

    @classmethod
    def start_monitoring(cls, interval_seconds: float):
        cls.monitor_stop_event.clear()
        cls.last_disk_io = None
        cls.last_disk_sample_time = None
        cls.monitor_thread = threading.Thread(
            target=cls._monitor_loop,
            args=(interval_seconds,),
            daemon=True,
            name="metrics-monitor"
        )
        cls.monitor_thread.start()

    @classmethod
    def stop_monitoring(cls):
        cls.monitor_stop_event.set()
        if cls.monitor_thread is not None and cls.monitor_thread.is_alive():
            cls.monitor_thread.join(timeout=2)
        cls.monitor_thread = None
        
    @classmethod
    def export_metrics_csv(cls, file_path: str):
        '''
        Exporta las métricas a un archivo CSV de cuatro columnas: inference_time, cpu_usage, ram_usage y disk_usage. Cada fila corresponde
        a una inferencia (en el caso de inference_time) o a una muestra del monitor (en el caso de cpu_usage, ram_usage y disk_usage).
        No existe relación entre las filas de inference_time y las filas de cpu_usage/ram_usage/disk_usage, ya que se muestrean de forma independiente.
        '''

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode='w', newline='') as csv_file:
            fieldnames = ['inference_time', 'cpu_usage', 'ram_usage', 'disk_usage']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            max_rows = max(
                len(cls.inference_latencies),
                len(cls.cpu_usage),
                len(cls.ram_usage),
                len(cls.disk_usage)
            )

            for i in range(max_rows):
                writer.writerow({
                    'inference_time': cls.inference_latencies[i] if i < len(cls.inference_latencies) else '',
                    'cpu_usage': cls.cpu_usage[i] if i < len(cls.cpu_usage) else '',
                    'ram_usage': cls.ram_usage[i] if i < len(cls.ram_usage) else '',
                    'disk_usage': cls.disk_usage[i] if i < len(cls.disk_usage) else ''
                })
                
        print(f"Métricas exportadas a {file_path}")