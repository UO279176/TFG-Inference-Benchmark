from pathlib import Path
import csv

from inference.metrics import Metrics


class MetricsP(Metrics):
    inference_TP = [] # Verdaderos positivos (TP) por cada inferencia
    inference_FP = [] # Falsos positivos (FP) por cada inferencia
    inference_FN = [] # Falsos negativos (FN) por cada inferencia
    
    @classmethod
    def _get_iou(cls, box_a, box_b):
        # Calcular el IoU entre dos cajas delimitadoras (área de intersección entre área de unión)
        # Formato de cajas: [ymin, xmin, ymax, xmax]
        ymin_i = max(box_a[0], box_b[0])
        xmin_i = max(box_a[1], box_b[1])
        ymax_i = min(box_a[2], box_b[2])
        xmax_i = min(box_a[3], box_b[3])

        inter_h = max(0.0, ymax_i - ymin_i)
        inter_w = max(0.0, xmax_i - xmin_i)
        inter_area = inter_h * inter_w # Área donde se superponen las dos cajas

        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area # Área total cubierta por ambas cajas

        return inter_area / union_area if union_area > 0 else 0.0

    @classmethod
    def _evaluate_bboxes(cls, gt_boxes, pred_boxes, pred_scores, iou_thresh=0.5):
        # Ordenar predicciones de mayor a menor confianza
        order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        gt_paired = set()
        tp, fp = 0, 0

        for idx in order: # Por cada caja predicha...
            pred = pred_boxes[idx]
            best_iou = 0.0
            best_gt = -1

            # Buscar la caja ground truth con el mayor IoU
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in gt_paired: # Ignorar cajas ground truth ya emparejadas
                    continue
                
                iou = cls._get_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            # Si tiene un IoU de 0.5 o más con alguna caja ground truth, la predicción es correcta
            if best_iou >= iou_thresh and best_gt != -1:
                tp += 1
                gt_paired.add(best_gt)
            else: # Si no, se ha inventado un humano donde no lo había
                fp += 1

        # Si al restar las cajas ground truth con las predicciones correctas nos queda alguna,
        # significa que no se han detectado humanos que sí estaban presentes
        fn = len(gt_boxes) - len(gt_paired)
        
        return tp, fp, fn
    
    @classmethod
    def add_inference_results(cls, sample, result):
        # Calcular TP, FP y FN
        res_TP, res_FP, res_FN = cls._evaluate_bboxes(sample.bboxes, result.locations, result.scores)
        
        cls.inference_TP.append(res_TP)
        cls.inference_FP.append(res_FP)
        cls.inference_FN.append(res_FN)
        
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
        print(f"Verdaderos positivos (TP): {sum(cls.inference_TP)}")
        print(f"Falsos positivos (FP): {sum(cls.inference_FP)}")
        print(f"Falsos negativos (FN): {sum(cls.inference_FN)}")
        print("=============================")
    
    @classmethod
    def export_metrics_csv(cls, file_path: str):
        '''
        Exporta las métricas a un archivo CSV de siete columnas: inference_time, TP, FP, FN, cpu_usage, ram_usage y disk_usage. Cada fila corresponde
        a una inferencia (en el caso de inference_time, TP, FP y FN) o a una muestra del monitor (en el caso de cpu_usage, ram_usage y disk_usage).
        No existe relación entre las filas de inference_time/TP/FP/FN y las filas de cpu_usage/ram_usage/disk_usage, ya que se muestrean de forma independiente.
        '''

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode='w', newline='') as csv_file:
            fieldnames = ['inference_time', 'TP', 'FP', 'FN', 'cpu_usage', 'ram_usage', 'disk_usage']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            max_rows = max(
                len(cls.inference_latencies),
                len(cls.inference_TP),
                len(cls.inference_FP),
                len(cls.inference_FN),
                len(cls.cpu_usage),
                len(cls.ram_usage),
                len(cls.disk_usage)
            )

            for i in range(max_rows):
                writer.writerow({
                    'inference_time': cls.inference_latencies[i] if i < len(cls.inference_latencies) else '',
                    'TP': cls.inference_TP[i] if i < len(cls.inference_TP) else '',
                    'FP': cls.inference_FP[i] if i < len(cls.inference_FP) else '',
                    'FN': cls.inference_FN[i] if i < len(cls.inference_FN) else '',
                    'cpu_usage': cls.cpu_usage[i] if i < len(cls.cpu_usage) else '',
                    'ram_usage': cls.ram_usage[i] if i < len(cls.ram_usage) else '',
                    'disk_usage': cls.disk_usage[i] if i < len(cls.disk_usage) else ''
                })
                
        print(f"Métricas exportadas a {file_path}")