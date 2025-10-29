"""
Generador de Subsets para Dataset Secuencial de Inventario

Este script permite generar subconjuntos del dataset basándose en el número 
de productos únicos que se desea incluir. Es útil para probar modelos de 
machine learning con diferentes cantidades de datos.

Autor: Dataset Slice Generator
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


class DatasetSliceGenerator:
    """
    Generador de subconjuntos (slices) de datasets secuenciales.
    """
    
    def __init__(self, dataset_path):
        """
        Inicializa el generador con la ruta al dataset completo.
        
        Args:
            dataset_path (str): Ruta al archivo CSV del dataset
        """
        self.dataset_path = Path(dataset_path)
        self.df_full = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Carga el dataset completo."""
        print(f"Cargando dataset desde: {self.dataset_path}")
        self.df_full = pd.read_csv(self.dataset_path)
        print(f"✓ Dataset cargado")
        print(f"  - Total de filas: {len(self.df_full):,}")
        print(f"  - Productos únicos: {self.df_full['product_id'].nunique():,}")
    
    def get_stats(self):
        """
        Obtiene estadísticas del dataset.
        
        Returns:
            dict: Diccionario con estadísticas del dataset
        """
        records_per_product = self.df_full.groupby('product_id').size()
        
        stats = {
            'total_rows': len(self.df_full),
            'unique_products': self.df_full['product_id'].nunique(),
            'avg_records_per_product': records_per_product.mean(),
            'median_records_per_product': records_per_product.median(),
            'min_records_per_product': records_per_product.min(),
            'max_records_per_product': records_per_product.max()
        }
        
        return stats
    
    def generate_subset(self, num_productos, metodo='aleatorio', random_state=42):
        """
        Genera un subconjunto del dataset con un número específico de productos.
        
        Args:
            num_productos (int): Número de productos únicos a incluir
            metodo (str): Método de selección ('aleatorio', 'primeros', 
                         'mas_registros', 'menos_registros')
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            pd.DataFrame: Subset generado
        """
        total_productos = self.df_full['product_id'].nunique()
        
        if num_productos > total_productos:
            raise ValueError(
                f"El número de productos solicitado ({num_productos}) "
                f"excede el total disponible ({total_productos})"
            )
        
        # Seleccionar productos según el método
        if metodo == 'aleatorio':
            productos_unicos = self.df_full['product_id'].unique()
            np.random.seed(random_state)
            productos_seleccionados = np.random.choice(
                productos_unicos, size=num_productos, replace=False
            )
        
        elif metodo == 'primeros':
            productos_seleccionados = self.df_full['product_id'].unique()[:num_productos]
        
        elif metodo == 'mas_registros':
            records_per_product = self.df_full.groupby('product_id').size()
            records_per_product = records_per_product.sort_values(ascending=False)
            productos_seleccionados = records_per_product.head(num_productos).index.values
        
        elif metodo == 'menos_registros':
            records_per_product = self.df_full.groupby('product_id').size()
            records_per_product = records_per_product.sort_values(ascending=True)
            productos_seleccionados = records_per_product.head(num_productos).index.values
        
        else:
            raise ValueError(
                f"Método '{metodo}' no reconocido. "
                f"Usa: 'aleatorio', 'primeros', 'mas_registros', o 'menos_registros'"
            )
        
        # Filtrar el dataframe
        df_subset = self.df_full[
            self.df_full['product_id'].isin(productos_seleccionados)
        ].copy()
        
        # Resetear índice
        df_subset = df_subset.reset_index(drop=True)
        
        # Estadísticas
        print(f"\n✓ Subset generado exitosamente")
        print(f"  - Método de selección: {metodo}")
        print(f"  - Productos seleccionados: {num_productos}")
        print(f"  - Total de filas en subset: {len(df_subset):,}")
        print(f"  - Promedio de filas por producto: {len(df_subset)/num_productos:.1f}")
        
        return df_subset
    
    def save_subset(self, df_subset, output_path):
        """
        Guarda el subset en un archivo CSV.
        
        Args:
            df_subset (pd.DataFrame): Subset a guardar
            output_path (str): Ruta del archivo de salida
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_subset.to_csv(output_path, index=False)
        
        file_size_mb = output_path.stat().st_size / (1024**2)
        print(f"\n✓ Subset guardado en: {output_path}")
        print(f"  - Tamaño del archivo: {file_size_mb:.2f} MB")
        
        return output_path
    
    def generate_multiple_subsets(self, sizes, metodo='aleatorio', 
                                  output_dir='data', save=True):
        """
        Genera múltiples subsets con diferentes tamaños.
        
        Args:
            sizes (list): Lista con el número de productos para cada subset
            metodo (str): Método de selección
            output_dir (str): Directorio donde guardar los subsets
            save (bool): Si True, guarda cada subset en un archivo CSV
            
        Returns:
            dict: Diccionario con los subsets generados
        """
        subsets = {}
        output_dir = Path(output_dir)
        
        print(f"\nGenerando {len(sizes)} subsets con método '{metodo}'...")
        print("=" * 70)
        
        for size in sizes:
            print(f"\n>>> Subset con {size} productos:")
            df_subset = self.generate_subset(size, metodo=metodo)
            
            key = f"{size}_productos"
            subsets[key] = df_subset
            
            if save:
                output_path = output_dir / f"subset_{size}_productos.csv"
                self.save_subset(df_subset, output_path)
        
        print("\n" + "=" * 70)
        print(f"✓ {len(subsets)} subsets generados exitosamente")
        
        return subsets


def main():
    """Función principal para uso desde línea de comandos."""
    
    parser = argparse.ArgumentParser(
        description='Genera subsets del dataset basado en número de productos'
    )
    
    parser.add_argument(
        'dataset_path',
        help='Ruta al archivo CSV del dataset completo'
    )
    
    parser.add_argument(
        '-n', '--num-productos',
        type=int,
        required=True,
        help='Número de productos a incluir en el subset'
    )
    
    parser.add_argument(
        '-m', '--metodo',
        choices=['aleatorio', 'primeros', 'mas_registros', 'menos_registros'],
        default='aleatorio',
        help='Método de selección de productos (default: aleatorio)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Ruta del archivo de salida (CSV)'
    )
    
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria para reproducibilidad (default: 42)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Mostrar estadísticas del dataset'
    )
    
    args = parser.parse_args()
    
    # Crear generador
    generator = DatasetSliceGenerator(args.dataset_path)
    
    # Mostrar estadísticas si se solicita
    if args.stats:
        stats = generator.get_stats()
        print("\n" + "=" * 70)
        print("ESTADÍSTICAS DEL DATASET")
        print("=" * 70)
        for key, value in stats.items():
            print(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value:,}")
        print("=" * 70)
    
    # Generar subset
    df_subset = generator.generate_subset(
        num_productos=args.num_productos,
        metodo=args.metodo,
        random_state=args.seed
    )
    
    # Guardar si se especifica output
    if args.output:
        generator.save_subset(df_subset, args.output)
    else:
        print("\nNota: No se especificó archivo de salida. Use -o para guardar el subset.")


if __name__ == '__main__':
    main()
