class Metrics:
    total_inference_time = 0 # Tiempo de inferencia
    inference_latencies = [] # Latencia por inferencia
    cpu_usage = [] # Uso de CPU
    ram_usage = [] # Uso de RAM
    accelerator_usage = [] # Uso del acelerador (GPU/NPU/TPU)
    consumption_usage = [] # Consumo energético
    
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
    def add_accelerator_usage(cls, usage: float):
        cls.accelerator_usage.append(usage)
    
    # Promedio de uso del acelerador
    @classmethod
    def average_accelerator_usage(cls) -> float:
        if len(cls.accelerator_usage) == 0:
            return 0.0
        else:
            return sum(cls.accelerator_usage) / len(cls.accelerator_usage)
    
    # Uso máximo del acelerador
    @classmethod
    def max_accelerator_usage(cls) -> float:
        if len(cls.accelerator_usage) == 0:
            return 0.0
        else:
            return max(cls.accelerator_usage)



    @classmethod
    def add_consumption_usage(cls, usage: float):
        cls.consumption_usage.append(usage)

    # Promedio de consumo energético
    @classmethod
    def average_consumption_usage(cls) -> float:
        if len(cls.consumption_usage) == 0:
            return 0.0
        else:
            return sum(cls.consumption_usage) / len(cls.consumption_usage)
    
    # Consumo energético máximo
    @classmethod
    def max_consumption_usage(cls) -> float:
        if len(cls.consumption_usage) == 0:
            return 0.0
        else:
            return max(cls.consumption_usage)
    
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
        print(f"Promedio de uso de RAM: {cls.average_ram_usage():.2f}%")
        print(f"Uso máximo de RAM: {cls.max_ram_usage():.2f}%")
        print(f"Promedio de uso del acelerador: {cls.average_accelerator_usage():.2f}%")
        print(f"Uso máximo del acelerador: {cls.max_accelerator_usage():.2f}%")
        print(f"Promedio de consumo energético: {cls.average_consumption_usage():.2f} W")
        print(f"Consumo energético máximo: {cls.max_consumption_usage():.2f} W")