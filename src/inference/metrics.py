import threading
import psutil
import csv


class Metrics:
    total_inference_time = 0 # Tiempo de inferencia
    inference_latencies = [] # Latencia por inferencia
    cpu_usage = [] # Uso de CPU
    ram_usage = [] # Uso de RAM (MiB)
    
    monitor_thread = None
    monitor_stop_event = threading.Event()
    monitor_lock = threading.Lock()
    
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
        if len(cls.cpu_usage) == 0:
            return 0.0
        else:
            return sum(cls.cpu_usage) / len(cls.cpu_usage)

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
    
    @classmethod
    def sample_system_metrics(cls):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_used_mib = psutil.virtual_memory().used / (1024 * 1024)

        with cls.monitor_lock:
            cls.add_cpu_usage(float(cpu_percent))
            cls.add_ram_usage(float(ram_used_mib))

    @classmethod
    def _monitor_loop(cls, interval_seconds: float):
        while not cls.monitor_stop_event.is_set():
            cls.sample_system_metrics()
            if cls.monitor_stop_event.wait(timeout=interval_seconds):
                break

    @classmethod
    def start_monitoring(cls, interval_seconds: float):
        cls.monitor_stop_event.clear()
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
        Exporta las métricas a un archivo CSV de tres columnas: inference_time, cpu_usage y ram_usage. Cada fila corresponde
        a una inferencia (en el caso de inference_time) o a una muestra del monitor (en el caso de cpu_usage y ram_usage).
        No existe relación entre las filas de inference_time y las filas de cpu_usage/ram_usage, ya que se muestrean de forma independiente.
        '''
        
        with open(file_path, mode='w', newline='') as csv_file:
            fieldnames = ['inference_time', 'cpu_usage', 'ram_usage']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            max_rows = max(
                len(cls.inference_latencies),
                len(cls.cpu_usage),
                len(cls.ram_usage),
            )

            for i in range(max_rows):
                writer.writerow({
                    'inference_time': cls.inference_latencies[i] if i < len(cls.inference_latencies) else '',
                    'cpu_usage': cls.cpu_usage[i] if i < len(cls.cpu_usage) else '',
                    'ram_usage': cls.ram_usage[i] if i < len(cls.ram_usage) else ''
                })
                
        print(f"Métricas exportadas a {file_path}")